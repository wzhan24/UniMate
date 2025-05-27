from typing import Union

import torch

from e3nn import o3
from e3nn.util.jit import compile_mode

from ..radial_basis import BesselBasis
from ..cutoffs import PolynomialCutoff


@compile_mode("script")
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

    def forward(self, pos, edge_index, period_vec):
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_vec = edge_vec - period_vec
        edge_length = torch.linalg.norm(edge_vec, dim=1)
        edge_sh = self.sh(edge_vec)
        return edge_vec, edge_length, edge_sh


@compile_mode("script")
class RadialBasisEdgeEncoding(torch.nn.Module):
    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.irreps_out= o3.Irreps([(self.basis.num_basis, (0, 1))])

    def forward(self, edge_length):
        edge_length_embedded = (
            self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        )
        return edge_length_embedded
