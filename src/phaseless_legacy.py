"""Legacy implementation module for DSMDL reference-code port."""

from __future__ import annotations

from . import phaseless_author as _src

ForwardConfig = _src.AuthorForwardConfig
TrainingConfigLegacy = _src.AuthorTrainingConfig
UNet3Ab = _src.AuthorUNet3Ab

annulus_gen_rand = _src.annulus_gen_rand
select_incidence_indices = _src.select_incidence_indices
generate_forward_dataset = _src.generate_author_forward_dataset
load_mat_dataset = _src.load_author_mat_dataset
compute_inputs_from_fields = _src.compute_author_inputs_from_fields
compute_inputs_from_mat = _src.compute_author_inputs_from_mat
train_unet3ab = _src.train_author_unet3ab
save_checkpoint = _src.save_author_checkpoint
load_checkpoint = _src.load_author_checkpoint
save_summary = _src.save_author_summary
relative_l2_legacy = _src.author_relative_l2

__all__ = [
    "ForwardConfig",
    "TrainingConfigLegacy",
    "UNet3Ab",
    "annulus_gen_rand",
    "select_incidence_indices",
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
