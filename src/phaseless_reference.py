"""Neutral-name API for the MATLAB/DSMDL Python port."""

from __future__ import annotations

from . import phaseless_legacy as _impl


ForwardConfig = _impl.ForwardConfig
TrainingConfigLegacy = _impl.TrainingConfigLegacy
UNet3Ab = _impl.UNet3Ab

annulus_gen_rand = _impl.annulus_gen_rand
select_incidence_indices = _impl.select_incidence_indices
generate_forward_dataset = _impl.generate_forward_dataset
load_mat_dataset = _impl.load_mat_dataset
compute_inputs_from_fields = _impl.compute_inputs_from_fields
compute_inputs_from_mat = _impl.compute_inputs_from_mat
train_unet3ab = _impl.train_unet3ab
save_checkpoint = _impl.save_checkpoint
load_checkpoint = _impl.load_checkpoint
save_summary = _impl.save_summary
relative_l2_legacy = _impl.relative_l2_legacy

__all__ = [
    "ForwardConfig",
    "TrainingConfigLegacy",
    "UNet3Ab",
    "select_incidence_indices",
    "annulus_gen_rand",
    "generate_forward_dataset",
    "load_mat_dataset",
    "compute_inputs_from_fields",
    "compute_inputs_from_mat",
    "train_unet3ab",
    "save_checkpoint",
    "load_checkpoint",
    "save_summary",
    "relative_l2_legacy",
]
