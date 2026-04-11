"""
Tests de fumee pour l'architecture v3.

Execution:
    .venv/bin/python test_v3.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortical_column_v3 import (
    IDX_V3,
    ColumnEnsembleV3,
    ColumnV3,
    GridCellLocationModule,
    MiniColumnV3,
)


def test_grid_cell_location_changes() -> None:
    grid = GridCellLocationModule(n_modules=4, dt=0.1)
    before = grid.get_code().clone()
    grid.integrate(torch.tensor([1.0, 0.2]), dt=0.5)
    after = grid.get_code().clone()
    assert not torch.allclose(before, after), "Le code grid cell devrait changer apres integration"


def test_signed_prediction_errors() -> None:
    mc = MiniColumnV3(dt=0.1)
    state = torch.zeros(1, len(IDX_V3))
    bu_dominant = mc(state, s=torch.tensor([80.0]), mu=torch.tensor([5.0]))
    td_dominant = mc(state, s=torch.tensor([5.0]), mu=torch.tensor([80.0]))
    assert bu_dominant.eps_pos.item() > td_dominant.eps_pos.item()
    assert td_dominant.eps_neg.item() > bu_dominant.eps_neg.item()


def test_predictive_routing_gate() -> None:
    col = ColumnV3(n_mc=24, k=4, dt=0.1)
    col.pattern_memory.data.fill_(5.0)
    col.state[:, IDX_V3["L6"]] = 10.0
    out = col(torch.linspace(0.0, 50.0, steps=24), mu=torch.full((24,), 30.0))
    assert torch.all(out["gamma_gate"] < 0.5), "Une prediction forte doit reduire le passage gamma"


def test_astro_plasticity_changes_weight() -> None:
    mc = MiniColumnV3(dt=0.1)
    state = torch.zeros(1, len(IDX_V3))
    prev = mc.J[IDX_V3["Ls_pos"], IDX_V3["L4"]].item()
    out = mc.prospective_step(state, s=torch.tensor([90.0]), mu=torch.tensor([5.0]), n_inf=6, learn=True)
    new = mc.J[IDX_V3["Ls_pos"], IDX_V3["L4"]].item()
    assert torch.isfinite(out.glio_gain)
    assert new >= prev, "La plasticite devrait renforcer L4->Ls_pos dans ce regime de mismatch"


def test_ensemble_v3_forward() -> None:
    ensemble = ColumnEnsembleV3(n_col=3, n_mc=20, k=4, dt=0.1)
    s_list = [
        torch.linspace(0.0, 60.0, steps=20),
        torch.linspace(10.0, 70.0, steps=20),
        torch.linspace(20.0, 80.0, steps=20),
    ]
    velocity_list = [torch.tensor([0.2, 0.0]), torch.tensor([0.0, 0.2]), torch.tensor([0.1, 0.1])]
    out = ensemble(s_list, velocity_list=velocity_list)
    assert out["consensus"].shape[0] == 20
    assert torch.isfinite(out["free_energy"])


if __name__ == "__main__":
    torch.manual_seed(0)
    test_grid_cell_location_changes()
    print("test_grid_cell_location_changes: OK")
    test_signed_prediction_errors()
    print("test_signed_prediction_errors: OK")
    test_predictive_routing_gate()
    print("test_predictive_routing_gate: OK")
    test_astro_plasticity_changes_weight()
    print("test_astro_plasticity_changes_weight: OK")
    test_ensemble_v3_forward()
    print("test_ensemble_v3_forward: OK")
    print("Tous les tests v3 sont passes.")
