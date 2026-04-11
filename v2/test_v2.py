"""
Tests de fumee pour l'architecture v2.

Execution:
    python3 "Cortical column code v2/test_v2.py"
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortical_column_v2 import IDX, ColumnEnsembleV2, ColumnV2, MiniColumnV2


def test_p_ca_gate() -> None:
    mc = MiniColumnV2(dt=1.0, theta_ca=0.05)
    state = torch.zeros(1, len(IDX))

    quiet = mc(state, s=torch.tensor([10.0]), mu=torch.tensor([10.0]))
    mismatch = mc(state, s=torch.tensor([80.0]), mu=torch.tensor([5.0]))

    assert mismatch.p_ca.item() >= quiet.p_ca.item(), "P_Ca devrait augmenter avec un mismatch"
    assert mismatch.eps_pos.item() > quiet.eps_pos.item(), "PE+ devrait refleter le drive bottom-up"


def test_sdr_emergence() -> None:
    col = ColumnV2(n_mc=32, k=5, dt=1.0)
    s = torch.linspace(0.0, 100.0, steps=32)
    out = col(s)
    assert int(out["sdr"].sum().item()) == 5, "Le k-WTA doit activer exactement k minicolonnes"


def test_thalamocortical_loop() -> None:
    col = ColumnV2(n_mc=16, k=4, dt=1.0)
    s = torch.full((16,), 35.0)

    errors = []
    for _ in range(25):
        col(s)
        errors.append(col.thalamocortical_error(s).abs().mean().item())

    assert errors[-1] < errors[0], "La prediction L6 devrait mieux suivre l'entree au fil des cycles"


def test_ensemble_forward() -> None:
    ensemble = ColumnEnsembleV2(n_col=3, n_mc=24, k=4, dt=1.0)
    s_list = [
        torch.linspace(0.0, 60.0, steps=24),
        torch.linspace(5.0, 65.0, steps=24),
        torch.linspace(10.0, 70.0, steps=24),
    ]
    out = ensemble(s_list)
    assert out["consensus"].shape[0] == 24
    assert torch.isfinite(out["free_energy"])


if __name__ == "__main__":
    torch.manual_seed(0)
    test_p_ca_gate()
    print("test_p_ca_gate: OK")
    test_sdr_emergence()
    print("test_sdr_emergence: OK")
    test_thalamocortical_loop()
    print("test_thalamocortical_loop: OK")
    test_ensemble_forward()
    print("test_ensemble_forward: OK")
    print("Tous les tests v2 sont passes.")

