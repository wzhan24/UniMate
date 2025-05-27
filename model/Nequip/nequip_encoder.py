import torch
from torch_scatter import scatter
from e3nn.o3 import Irreps
from e3nn.util import jit
from model.Nequip.layers import embedding as e
from model.Nequip.layers import AtomwiseLinear
from model.Nequip.layers import ConvNetLayer, InteractionBlock
from utils import calculate_to_jimages_efficient, get_pbc_distances, get_pbc_cutoff_graphs, correct_cart_coords


class NequipEncoder(torch.nn.Module):
    def __init__(self,
            n_elems=100,
            conv_feature_size=32,
            parity = True,
            lmax = 1,
             cutoff=1.5,
             n_radial_basis=8,
             poly_degree=6,
             n_conv_layers=3,
             radial_network_hidden_dim=64,
             radial_network_layers=2,
             average_num_neigh=25,
             post_conv_layers=2,
             out_hidden_channels=16,
             out_channels=1
                 ):
        super().__init__()

        self.node_attr_layer = e.OneHotAtomEncoding(n_elems)
        node_attr_irrep = self.node_attr_layer.irreps_out


        node_feature_irrep = Irreps([(conv_feature_size, (0, 1))])
        self.node_embedding_layer = AtomwiseLinear(node_attr_irrep, node_feature_irrep)

        p = -1 if parity else 1
        edge_attr_irrep = Irreps([(1,(l,p**l)) for l in range(lmax + 1)])
        self.edge_attr_layer  = e.SphericalHarmonicEdgeAttrs(edge_attr_irrep)
        assert (edge_attr_irrep == self.edge_attr_layer.irreps_out)

        cutoff_kwargs = {"r_max": cutoff, "p": poly_degree}
        basis_kwargs = {"r_max": cutoff, "num_basis": n_radial_basis}
        radial_basis_layer = e.RadialBasisEdgeEncoding(basis_kwargs=basis_kwargs, cutoff_kwargs=cutoff_kwargs)
        edge_feature_irrep = radial_basis_layer.irreps_out
        self.edge_embedding_layer = radial_basis_layer


        # convolution layers
        node_feature_irrep_intermidiate = []
        conv_hidden_irrep = Irreps(
            [(conv_feature_size, (l, p)) for p in ((1, -1) if parity else (1,)) for l in
             range(lmax + 1)])
        invariant_neurons = radial_network_hidden_dim
        invariant_layers = radial_network_layers
        average_num_neigh =average_num_neigh
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
        post_conv_layers_list = []
        out_cs = [out_hidden_channels,out_channels]
        assert len(out_cs) == post_conv_layers
        for i in range(post_conv_layers):
            linear_post_layer = AtomwiseLinear(last_node_irrep, str(out_cs[i])+'x0e')
            post_conv_layers_list.append(linear_post_layer)
            last_node_irrep = linear_post_layer.irreps_out
        self.post_conv_layers = torch.nn.ModuleList(post_conv_layers_list)

        self.out_norm = torch.nn.LayerNorm(out_channels)

    def forward(self,batch_data, construct_graph=False):
        z, edge_index, frac_coords, batch = batch_data.node_feat, batch_data.edge_index, batch_data.frac_coords, batch_data.batch
        cart_coords = batch_data.cart_coords
        num_atoms, num_bonds = batch_data.num_atoms, batch_data.num_edges
        lattice_lengths, lattice_angles = batch_data.lengths, batch_data.angles
        i, j = edge_index
        # elem/node embedding


        x_attr = self.node_attr_layer(z.long().squeeze(-1))
        x_attr = x_attr.to(dtype=cart_coords.dtype)
        # Linear map for embedding
        h = self.node_embedding_layer(x_attr)

        if hasattr(batch_data, 'to_jimages'):
            to_jimages = batch_data.to_jimages
        else:
            to_jimages = calculate_to_jimages_efficient(cart_coords, edge_index, batch_data.lattice_vectors, num_bonds)

        # cart_coords = correct_cart_coords(cart_coords, lattice_lengths, lattice_angles, num_atoms, batch)
        # edge_index, distance_vectors, pbc_offset = get_pbc_cutoff_graphs(cart_coords, lattice_lengths, lattice_angles,
        #                                                                   num_atoms, 2.0,9)

        _, _, pbc_offset = get_pbc_distances(cart_coords, edge_index, lattice_lengths, lattice_angles,
                                                   to_jimages, num_atoms, num_bonds, coord_is_cart=True)
        # Edge embedding
        edge_vec, edge_lengths, edge_sh = self.edge_attr_layer(cart_coords, edge_index, pbc_offset)


        # Radial basis function
        edge_length_embedding = self.edge_embedding_layer(edge_lengths)

        # convolution
        for layer in self.conv_layers:
            h = layer(x_attr, h, edge_length_embedding, edge_sh, edge_index)

        # post convolution
        for layer in self.post_conv_layers:
            h = layer(h)

        h = h.squeeze()
        h = self.out_norm(h)

        # reduce to graph representation
        h = scatter(h, batch, dim=0)

        return h