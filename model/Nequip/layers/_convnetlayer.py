from typing import Dict, Callable
import torch
import logging

from e3nn import o3
from e3nn.nn import Gate, NormActivation
import math
from . import InteractionBlock

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

#@torch.jit.script # removing scripting to avoid the Nan bug
def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)

acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}


class ConvNetLayer(torch.nn.Module):
    """
    Args:

    """

    resnet: bool

    def __init__(
        self,
        irreps_in, # its node embedding
        feature_irreps_hidden,
        node_attr_irreps,
        edge_attr_irreps,
        edge_embedding_irreps,        
        convolution_kwargs: dict = {},
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()
        # initialization
        assert nonlinearity_type in ("gate", "norm")
        # make the nonlin dicts from parity ints instead of convinience strs
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.feature_irreps_hidden = feature_irreps_hidden
        self.resnet = resnet

        self.irreps_in = irreps_in

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0
                and tp_path_exists(irreps_in, edge_attr_irreps, ir)
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0
                and tp_path_exists(irreps_in, edge_attr_irreps, ir)
            ]
        )

        irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = (
                "0e"
                if tp_path_exists(irreps_in, edge_attr_irreps, "0e")
                else "0o"
            )
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            # TO DO, it's not that safe to directly use the
            # dictionary
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            conv_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            conv_irreps_out = irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=conv_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=acts[nonlinearity_scalars[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin

        # TODO: partial resnet?
        if irreps_layer_out == irreps_in and resnet:
            # We are doing resnet updates and can for this layer
            self.resnet = True
        else:
            self.resnet = False

        # override defaults for irreps:
        self.conv = InteractionBlock(
            irreps_in=self.irreps_in,
            irreps_out=conv_irreps_out,
            node_attr_irreps=node_attr_irreps,
            edge_attr_irreps=edge_attr_irreps,
            edge_embedding_irreps=edge_embedding_irreps,
            **convolution_kwargs
        )
        # The output features are whatever we got in
        # updated with whatever the convolution outputs (which is a full graph module)
        self.irreps_out = self.equivariant_nonlin.irreps_out

    def forward(self, x, h, edge_length_embeddings, edge_sh, edge_index):
        # save old features for resnet
        old_h = h
        # run convolution
        h = self.conv(x, h, edge_length_embeddings, edge_sh, edge_index)
        # do nonlinearity
        h = self.equivariant_nonlin(h)

        # do resnet
        if self.resnet:
            h = old_h + h
        return h
