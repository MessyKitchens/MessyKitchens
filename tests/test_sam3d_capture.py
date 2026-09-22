"""CPU mechanism tests for observational hooks; these do not replace GPU smoke."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from multi_object_decoder.sam3d_capture import (
    SAM3DCacheCapture,
    tensor_preserving_asdict,
)


class Solver:
    def step(self, dynamics, x_t, t, dt, *args, **kwargs):
        return x_t + dt * dynamics(x_t, t)


class Backbone:
    def split_latent_share_transformer(self, value):
        return {"translation": value}


class Pipeline:
    def __init__(self):
        self.backbone = Backbone()
        self.solver = Solver()
        self.models = {"ss_generator": SimpleNamespace(
            reverse_fn=SimpleNamespace(backbone=self.backbone),
            _solver=self.solver,
            _solver_method="euler",
        )}

    def get_condition_input(self, value):
        return (value,), {}

    def sample_sparse_structure(self, inputs, fail=False):
        self.get_condition_input(torch.ones(1, 2, 3))
        x_t = inputs["initial"]
        for t in (0.0, 0.5):
            def dynamics(value, time):
                self.backbone.split_latent_share_transformer(value + time)
                return value + 2.0
            x_t = self.solver.step(dynamics, x_t, t, 0.5)
        if fail:
            raise RuntimeError("controlled sampling failure")
        return x_t


def _inputs():
    return {
        "pointmap_scale": torch.ones(1, 3),
        "pointmap_shift": torch.zeros(1, 3),
        "initial": torch.zeros(1, 1, 3),
    }


def test_observer_keeps_result_and_captures_final_original_step(tmp_path):
    pipeline = Pipeline()
    inputs = _inputs()
    expected = pipeline.sample_sparse_structure(inputs)
    original_methods = (
        pipeline.sample_sparse_structure, pipeline.get_condition_input,
        pipeline.solver.step, pipeline.backbone.split_latent_share_transformer,
    )
    capture = SAM3DCacheCapture(pipeline)
    with capture.observe():
        actual = pipeline.sample_sparse_structure(inputs)
    assert torch.equal(actual, expected)
    assert torch.equal(inputs["initial"], torch.zeros(1, 1, 3))
    assert capture.payload["solver_state"] == {"t": 0.5, "dt": 0.5}
    assert torch.equal(capture.payload["x_t_last_step"], torch.ones(1, 1, 3))
    assert torch.equal(
        capture.payload["pose_latents_raw"]["translation"], torch.full((1, 1, 3), 1.5)
    )
    assert original_methods == (
        pipeline.sample_sparse_structure, pipeline.get_condition_input,
        pipeline.solver.step, pipeline.backbone.split_latent_share_transformer,
    )
    capture.save(tmp_path, "003")
    with np.load(tmp_path / "obj_003.npz", allow_pickle=True) as archive:
        assert archive["solver_state"].item() == {"t": 0.5, "dt": 0.5}
        assert np.array_equal(archive["x_t_last_step"], np.ones((1, 1, 3)))
    assert np.load(tmp_path / "img_cond_003.npy", allow_pickle=False).shape == (1, 2, 3)


def test_observer_restores_every_method_after_sampling_error(tmp_path):
    pipeline = Pipeline()
    before = {id(owner): set(vars(owner)) for owner in (
        pipeline, pipeline.solver, pipeline.backbone
    )}
    capture = SAM3DCacheCapture(pipeline)
    with pytest.raises(RuntimeError, match="controlled sampling failure"):
        with capture.observe():
            pipeline.sample_sparse_structure(_inputs(), fail=True)
    for owner in (pipeline, pipeline.solver, pipeline.backbone):
        assert set(vars(owner)) == before[id(owner)]
    with pytest.raises(RuntimeError, match="successful SS sampling"):
        capture.save(tmp_path, "000")
    assert not list(tmp_path.iterdir())


def test_observer_rejects_non_euler_solver_before_sampling():
    pipeline = Pipeline()
    pipeline.models["ss_generator"]._solver_method = "midpoint"
    with pytest.raises(RuntimeError, match="Euler"):
        with SAM3DCacheCapture(pipeline).observe():
            pass


def test_tensor_preserving_dataclass_conversion_keeps_nonleaf_gradient():
    @dataclass
    class Pose:
        translation: torch.Tensor
        metadata: dict

    original = torch.tensor([2.0], requires_grad=True)
    nonleaf = original * 3.0
    pose = Pose(nonleaf, {"values": [1, 2]})
    output = tensor_preserving_asdict(pose)
    assert output["translation"] is nonleaf
    assert output["metadata"] == pose.metadata
    assert output["metadata"] is not pose.metadata
    output["translation"].sum().backward()
    assert original.grad.item() == 3.0
