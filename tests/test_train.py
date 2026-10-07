"""Checkpointing must not change the vanilla recipe, and resuming must continue it exactly."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.train import TrainConfig, train


def _setup():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, 2, 8, 8, generator=g)
    y = 0.5 * x + 0.1 * torch.randn(40, 2, 8, 8, generator=g)
    cfg = TrainConfig(epochs=4, batch_size=8, device="cpu")
    return x, y, cfg


def _model():
    torch.manual_seed(7)
    return torch.nn.Conv2d(2, 2, 3, padding=1)


def test_callback_does_not_change_training():
    x, y, cfg = _setup()
    _, plain = train(_model(), x, y, cfg)
    seen = []
    _, hooked = train(_model(), x, y, cfg, on_epoch_end=lambda e, *a: seen.append(e))
    assert hooked == plain
    assert seen == [0, 1, 2, 3]


def test_resume_reproduces_an_uninterrupted_run():
    x, y, cfg = _setup()
    _, full = train(_model(), x, y, cfg)

    saved = {}

    def save(epoch, model, opt, sched, history):
        if epoch == 1:
            saved.update(copy.deepcopy({"model": model.state_dict(), "opt": opt.state_dict(),
                                        "sched": sched.state_dict(), "history": history,
                                        "rng": torch.get_rng_state()}))

    train(_model(), x, y, cfg, on_epoch_end=save)
    _, resumed = train(_model(), x, y, cfg, resume=saved)
    assert len(resumed) == 4
    for a, b in zip(resumed, full):
        assert a["epoch"] == b["epoch"]
        assert abs(a["train"] - b["train"]) <= 1e-6 * abs(b["train"])
        assert abs(a["val"] - b["val"]) <= 1e-6 * abs(b["val"])
