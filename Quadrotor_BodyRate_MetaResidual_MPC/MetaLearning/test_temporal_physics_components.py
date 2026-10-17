"""Smoke tests for temporal physics-disentangled meta components."""

import sys
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bodyrate_common import (
    CONTEXT_COLS,
    MODEL_INPUT_DIM,
    RESIDUAL_OUTPUT_DIM,
    PhysicsDisentangledContextResidualMLP,
    GroupedTemporalPhysicsSupportSetEncoder,
)


def main():
    torch.manual_seed(7)
    model = PhysicsDisentangledContextResidualMLP(hidden_dim=32, num_layers=2)
    encoder = GroupedTemporalPhysicsSupportSetEncoder(hidden_dim=24, num_layers=2)
    features = torch.randn(24, MODEL_INPUT_DIM)
    targets = torch.randn(24, RESIDUAL_OUTPUT_DIM)
    context = encoder(features, targets)
    assert context.shape == (1, len(CONTEXT_COLS))

    x = torch.randn(5, MODEL_INPUT_DIM)
    base_context = torch.randn(5, len(CONTEXT_COLS))
    output = model.forward_with_context(x, base_context)
    assert output.shape == (5, RESIDUAL_OUTPUT_DIM)
    output.square().mean().backward()

    # Rotational context must not affect translational residual channels.
    changed_rotation = base_context.clone()
    changed_rotation[:, [1, 2, 3, 5]] += 3.0
    assert torch.allclose(
        output[:, :3], model.forward_with_context(x, changed_rotation)[:, :3], atol=1e-6
    )
    # Mass/thrust/aerodynamic context must not affect angular-rate residuals.
    changed_translation = base_context.clone()
    changed_translation[:, [0, 4, 6, 7, 8, 9, 10]] -= 2.0
    assert torch.allclose(
        output[:, 3:], model.forward_with_context(x, changed_translation)[:, 3:], atol=1e-6
    )
    print("temporal physics-disentangled component checks passed")


if __name__ == "__main__":
    main()
