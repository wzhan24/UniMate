import logging
from typing import Optional, List

import torch
import torch.nn.functional

from e3nn.o3 import Linear


# class AtomwiseOperation(torch.nn.Module):
#     def __init__(self, operation, field: str, irreps_in=None):
#         super().__init__()
#         self.operation = operation
#         # self.field = field
#         self._init_irreps(
#             irreps_in=irreps_in,
#             my_irreps_in={field: operation.irreps_in},
#             irreps_out={field: operation.irreps_out},
#         )

#     def forward(self, data):
#         data = self.operation(data)
#         return data


class AtomwiseLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
    ):
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        super().__init__()
        self.linear = Linear(
            irreps_in=self.irreps_in, irreps_out=self.irreps_out
        )

    def forward(self, h):
        h = self.linear(h)
        return h


# class AtomwiseReduce(torch.nn.Module):
#     constant: float

#     def __init__(
#         self,
#         field: str,
#         out_field: Optional[str] = None,
#         reduce="sum",
#         avg_num_atoms=None,
#         irreps_in={},
#     ):
#         super().__init__()
#         assert reduce in ("sum", "mean", "normalized_sum")
#         self.constant = 1.0
#         if reduce == "normalized_sum":
#             assert avg_num_atoms is not None
#             self.constant = float(avg_num_atoms) ** -0.5
#             reduce = "sum"
#         self.reduce = reduce
#         self.field = field
#         self.out_field = f"{reduce}_{field}" if out_field is None else out_field
#         self._init_irreps(
#             irreps_in=irreps_in,
#             irreps_out={self.out_field: irreps_in[self.field]}
#             if self.field in irreps_in
#             else {},
#         )

#     def forward(self, h, contrib):
#         E = scatter(h, contrib, dim=0, reduce=self.reduce)
#         if self.constant != 1.0:
#             E = E * self.constant
#         return E


# class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
#     """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

#     Args:
#         field: the per-atom field to scale/shift.
#         num_types: the number of types in the model.
#         shifts: the initial shifts to use, one per atom type.
#         scales: the initial scales to use, one per atom type.
#         arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
#             units (in which case they will be rescaled appropriately by any global rescaling later
#             applied to the model); if ``False``, the provided shifts/scales will be used without modification.

#             For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
#             But if scales/shifts computed from the training data are used, and are thus in dataset units,
#             this should be ``True``.
#         out_field: the output field; defaults to ``field``.
#     """

#     field: str
#     out_field: str
#     scales_trainble: bool
#     shifts_trainable: bool
#     has_scales: bool
#     has_shifts: bool

#     def __init__(
#         self,
#         field: str,
#         num_types: int,
#         type_names: List[str],
#         shifts: Optional[List[float]],
#         scales: Optional[List[float]],
#         arguments_in_dataset_units: bool,
#         out_field: Optional[str] = None,
#         scales_trainable: bool = False,
#         shifts_trainable: bool = False,
#         irreps_in={},
#     ):
#         super().__init__()
#         self.num_types = num_types
#         self.type_names = type_names
#         self.field = field
#         self.out_field = f"shifted_{field}" if out_field is None else out_field
#         self._init_irreps(
#             irreps_in=irreps_in,
#             my_irreps_in={self.field: "0e"},  # input to shift must be a single scalar
#             irreps_out={self.out_field: irreps_in[self.field]},
#         )

#         self.has_shifts = shifts is not None
#         if shifts is not None:
#             shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
#             if len(shifts.reshape([-1])) == 1:
#                 shifts = torch.ones(num_types) * shifts
#             assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
#             self.shifts_trainable = shifts_trainable
#             if shifts_trainable:
#                 self.shifts = torch.nn.Parameter(shifts)
#             else:
#                 self.register_buffer("shifts", shifts)

#         self.has_scales = scales is not None
#         if scales is not None:
#             scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
#             if len(scales.reshape([-1])) == 1:
#                 scales = torch.ones(num_types) * scales
#             assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
#             self.scales_trainable = scales_trainable
#             if scales_trainable:
#                 self.scales = torch.nn.Parameter(scales)
#             else:
#                 self.register_buffer("scales", scales)

#         self.arguments_in_dataset_units = arguments_in_dataset_units

#     def forward(self, x, h):

#         if not (self.has_scales or self.has_shifts):
#             return h

#         assert len(h) == len(x), "h doesnt seem to have correct per-atom shape"
#         if self.has_scales:
#             h = self.scales[x].view(-1, 1) * h
#         if self.has_shifts:
#             h = self.shifts[x].view(-1, 1) + h
#         return h

#     def update_for_rescale(self, rescale_module):
#         if hasattr(rescale_module, "related_scale_keys"):
#             if self.out_field not in rescale_module.related_scale_keys:
#                 return
#         if self.arguments_in_dataset_units and rescale_module.has_scale:
#             logging.debug(
#                 f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
#                 f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
#                 f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
#             )
#             with torch.no_grad():
#                 if self.has_scales:
#                     self.scales.div_(rescale_module.scale_by)
#                 if self.has_shifts:
#                     self.shifts.div_(rescale_module.scale_by)
#             logging.debug(
#                 f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
#                 f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
#             )
