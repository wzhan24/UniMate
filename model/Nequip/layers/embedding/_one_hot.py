import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode



@compile_mode("script")
class OneHotAtomEncoding(torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    def __init__(
        self,
        num_elems: int,
        irreps_in=None,
    ):
        super().__init__()
        self.num_elems = num_elems
        # Output irreps are num_elems even (invariant) scalars
        self.irreps_in = irreps_in
        self.irreps_out = Irreps([(self.num_elems, (0, 1))])

    def forward(self, x): #TODO input data type
        one_hot = torch.nn.functional.one_hot(x, num_classes=self.num_elems)
        return one_hot
