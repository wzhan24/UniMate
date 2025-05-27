import torch
from e3nn.nn import FullyConnectedNet
import math
from torch_scatter import scatter
from e3nn.o3 import Irreps, TensorProduct
from e3nn.util import jit
from model.Nequip.layers import embedding as e
from model.Nequip.layers import AtomwiseLinear
from model.Nequip.layers import ConvNetLayer, InteractionBlock
from utils import calculate_to_jimages_efficient, get_pbc_distances
from e3nn import o3
from typing import Dict, Callable, Union


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)

class SphericalHarmonicEdgeAttrs(torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    def __init__(
        self,
        irreps_in: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True):
        super().__init__()

        if isinstance(irreps_in, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_in)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_in)
        self.irreps_in=irreps_in
        self.irreps_out=self.irreps_edge_sh
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, edge_vec):
        edge_sh = self.sh(edge_vec)
        return edge_sh





class NequipDecoder(torch.nn.Module):
    def __init__(self,
            n_elems=100,
            hidden_channels=32,
            parity = True,
            lmax = 1,
             cutoff=1.5,
             n_radial_basis=8,
             poly_degree=6,
             n_conv_layers=3,
             radial_network_hidden_dim=64,
             radial_network_layers=2,
             average_num_neigh=25,
             nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"}
                 ):
        super().__init__()

        self.node_attr_layer = e.OneHotAtomEncoding(n_elems)
        node_attr_irrep = self.node_attr_layer.irreps_out


        node_feature_irrep = Irreps([(hidden_channels, (0, 1))])
        self.node_embedding_layer = AtomwiseLinear(node_attr_irrep, node_feature_irrep)

        p = -1 if parity else 1
        edge_attr_irrep = Irreps([(1,(l,p**l)) for l in range(lmax + 1)])
        self.edge_attr_layer  = SphericalHarmonicEdgeAttrs(edge_attr_irrep)
        assert (edge_attr_irrep == self.edge_attr_layer.irreps_out)

        cutoff_kwargs = {"r_max": cutoff, "p": poly_degree}
        basis_kwargs = {"r_max": cutoff, "num_basis": n_radial_basis}
        radial_basis_layer = e.RadialBasisEdgeEncoding(basis_kwargs=basis_kwargs, cutoff_kwargs=cutoff_kwargs)
        edge_feature_irrep = radial_basis_layer.irreps_out
        self.edge_embedding_layer = radial_basis_layer


        # convolution layers
        node_feature_irrep_intermidiate = []
        conv_hidden_irrep = Irreps(
            [(hidden_channels, (l, p)) for p in ((1, -1) if parity else (1,)) for l in
             range(lmax + 1)])
        invariant_neurons = radial_network_hidden_dim
        invariant_layers = radial_network_layers
        average_num_neigh = average_num_neigh
        conv_kw = {"invariant_layers": invariant_layers, "invariant_neurons": invariant_neurons,
                   "avg_num_neighbors": average_num_neigh}

        last_node_irrep = node_feature_irrep
        conv_layers = []
        for i in range(n_conv_layers):
            conv_layer = ConvNetLayer(last_node_irrep,
                                      conv_hidden_irrep,
                                      node_attr_irrep,
                                      edge_attr_irrep,
                                      edge_feature_irrep,
                                      convolution_kwargs=conv_kw)
            conv_layers.append(conv_layer)
            last_node_irrep = conv_layer.irreps_out
        self.conv_layers = torch.nn.ModuleList(conv_layers)

        # final mappings
        irreps_mid = []
        instructions = []

        for i, (mul, ir_in) in enumerate(last_node_irrep):
            for j, (_, ir_edge) in enumerate(edge_attr_irrep):
                for ir_out in ir_in * ir_edge:
                    if ir_out in edge_attr_irrep:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            last_node_irrep,
            edge_attr_irrep,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )


        self.fc = FullyConnectedNet(
            [edge_feature_irrep.num_irreps]
            + invariant_layers * [invariant_neurons]
            + [tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.tp = tp

        self.lin =  o3.Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=node_feature_irrep,
            internal_weights=True,
            shared_weights=True,
        )
        self.layer_norm = torch.nn.LayerNorm(last_node_irrep.dim)

    def forward(self,z, edge_index, distance_vectors):
        j,i = edge_index
        x_attr = self.node_attr_layer(z.long().squeeze(-1))
        x_attr = x_attr.to(dtype=distance_vectors.dtype)
        # Linear map for embedding
        h = self.node_embedding_layer(x_attr)

        # Edge embedding
        edge_sh = self.edge_attr_layer(distance_vectors)
        edge_lengths = torch.linalg.norm(distance_vectors, dim=1)


        # Radial basis function
        edge_length_embedding = self.edge_embedding_layer(edge_lengths)

        # convolution
        for layer in self.conv_layers:
            h = layer(x_attr, h, edge_length_embedding, edge_sh, edge_index)

        h = h.squeeze()
        h = self.layer_norm(h)

        # compute edge_feature
        edge_weight = self.fc(edge_length_embedding)
        edge_features = self.tp(h[i], edge_sh, edge_weight)
        edge_features = self.lin(edge_features)


        return edge_features