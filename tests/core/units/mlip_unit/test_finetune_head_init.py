"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from fairchem.core.units.mlip_unit.mlip_unit import (
    initialize_finetuning_model,
    load_inference_model,
)


def _checkpoint_head(direct_checkpoint):
    inference_ckpt, _ = direct_checkpoint
    model, checkpoint = load_inference_model(inference_ckpt)
    name = sorted(k for k in model.output_heads if k.endswith("energy"))[0]
    config = deepcopy(dict(checkpoint.model_config["heads"][name]))
    return inference_ckpt, name, config, model.output_heads[name].state_dict()


def test_head_init_from_copies_checkpoint_head(direct_checkpoint):
    inference_ckpt, name, config, source_state = _checkpoint_head(direct_checkpoint)

    model = initialize_finetuning_model(
        inference_ckpt, heads={"new": config}, head_init_from={"new": name}
    )

    target_state = model.output_heads["new"].state_dict()
    assert target_state.keys() == source_state.keys()
    for key, value in source_state.items():
        assert torch.equal(target_state[key], value)


def test_head_init_from_omitted_leaves_fresh_head(direct_checkpoint):
    inference_ckpt, _, config, source_state = _checkpoint_head(direct_checkpoint)

    torch.manual_seed(1)
    model = initialize_finetuning_model(inference_ckpt, heads={"new": config})

    target_state = model.output_heads["new"].state_dict()
    assert any(
        not torch.equal(target_state[key], value) for key, value in source_state.items()
    )


def test_head_init_from_rejects_mismatched_heads(direct_checkpoint):
    inference_ckpt, name, config, _ = _checkpoint_head(direct_checkpoint)
    mismatched = {"module": "fairchem.core.models.uma.escn_md.Linear_Force_Head"}

    with pytest.raises(RuntimeError, match="state_dict"):
        initialize_finetuning_model(
            inference_ckpt,
            heads={"new": config, "other": mismatched},
            head_init_from={"other": name},
        )
