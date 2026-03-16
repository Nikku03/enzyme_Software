from __future__ import annotations

import pytest


def test_site_loss_ignores_unsupervised_nodes():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.training.loss import SiteOfMetabolismLoss

    logits = torch.tensor([[2.0], [-2.0], [6.0], [-6.0]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    supervision_mask = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)

    loss_fn = SiteOfMetabolismLoss()
    loss, stats = loss_fn(logits, labels, batch, supervision_mask=supervision_mask)

    assert bool(torch.isfinite(loss))
    assert stats["site_loss"] >= 0.0


def test_site_metrics_ignore_unsupervised_molecules():
    from enzyme_software.liquid_nn_v2.training.metrics import compute_site_metrics_v2
    import numpy as np

    scores = np.asarray([[0.9], [0.1], [0.99], [0.98]], dtype=np.float32)
    labels = np.asarray([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    batch = np.asarray([0, 0, 1, 1], dtype=np.int64)
    supervision_mask = np.asarray([[1.0], [1.0], [0.0], [0.0]], dtype=np.float32)

    metrics = compute_site_metrics_v2(scores, labels, batch, supervision_mask=supervision_mask)

    assert metrics["site_top1_acc"] == 1.0
    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 0.0
