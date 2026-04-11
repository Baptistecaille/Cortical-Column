"""
Architecture v2 de colonne corticale.

Cette version traduit la refonte décrite dans
`03 - Concepts/architecture-v2-repensee.md` tout en intégrant deux briques
prioritaires de la roadmap v3 :
  - erreurs de prediction signees PE+/PE-
  - plasticite synaptique a court terme (STP) sur quelques boucles inhibitrices

Le but est de fournir une base de code simple, executable et extensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


POPULATIONS = ["L4", "Ls_pos", "Ls_neg", "Ld", "L6", "PV", "SST", "VIP"]
IDX: Dict[str, int] = {name: i for i, name in enumerate(POPULATIONS)}
N_STATE = len(POPULATIONS)

TAU = torch.tensor([15.0, 20.0, 20.0, 25.0, 40.0, 5.0, 30.0, 15.0])
R_MAX = 100.0


def build_biological_J() -> torch.Tensor:
    """Connectivite initiale inspiree du microcircuit canonique."""
    J = torch.zeros(N_STATE, N_STATE)

    # Excitation
    J[IDX["L4"], IDX["L4"]] = 0.30
    J[IDX["Ls_pos"], IDX["L4"]] = 1.40
    J[IDX["Ls_neg"], IDX["L4"]] = 0.90
    J[IDX["Ld"], IDX["Ls_pos"]] = 0.80
    J[IDX["Ld"], IDX["Ls_neg"]] = 0.80
    J[IDX["L6"], IDX["Ls_pos"]] = 0.35
    J[IDX["L6"], IDX["Ls_neg"]] = 0.35
    J[IDX["L6"], IDX["Ld"]] = 0.25
    J[IDX["L6"], IDX["L6"]] = 0.45
    J[IDX["PV"], IDX["L4"]] = 0.90
    J[IDX["PV"], IDX["Ls_pos"]] = 0.40
    J[IDX["PV"], IDX["Ls_neg"]] = 0.40
    J[IDX["SST"], IDX["Ld"]] = 0.45

    # Inhibition
    J[IDX["Ls_pos"], IDX["PV"]] = -0.80
    J[IDX["Ls_neg"], IDX["PV"]] = -0.60
    J[IDX["Ls_pos"], IDX["SST"]] = -0.35
    J[IDX["Ls_neg"], IDX["SST"]] = -0.35
    J[IDX["Ld"], IDX["PV"]] = -0.45
    J[IDX["Ld"], IDX["SST"]] = -0.30
    J[IDX["L4"], IDX["L6"]] = -0.25
    J[IDX["SST"], IDX["VIP"]] = -0.80
    J[IDX["PV"], IDX["VIP"]] = -0.20

    return J


class STPSynapse:
    """Synapse inhibitrice avec modele de Tsodyks-Markram."""

    def __init__(
        self,
        mode: str = "STD",
        tau_d: float = 200.0,
        tau_f: float = 500.0,
        U: float = 0.5,
        dt: float = 1.0,
    ):
        self.mode = mode
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.U = U
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.x = torch.tensor(1.0)
        self.u = torch.tensor(self.U)

    def update(self, r_pre: torch.Tensor) -> None:
        r_pre = torch.as_tensor(r_pre).float().mean()
        dx = (1.0 - self.x) / self.tau_d - self.u * self.x * (r_pre / R_MAX)
        if self.mode == "STF":
            du = (self.U - self.u) / self.tau_f + self.U * (1.0 - self.u) * (r_pre / R_MAX)
        else:
            du = (self.U - self.u) / self.tau_f
        self.x = torch.clamp(self.x + self.dt * dx, 0.0, 1.0)
        self.u = torch.clamp(self.u + self.dt * du, 0.0, 1.5)

    def effective_weight(self, base_weight: torch.Tensor, r_pre: torch.Tensor) -> torch.Tensor:
        self.update(r_pre)
        return base_weight * self.x * self.u


@dataclass
class MiniColumnOutput:
    state: torch.Tensor
    surprise: torch.Tensor
    eps_pos: torch.Tensor
    eps_neg: torch.Tensor
    p_ca: torch.Tensor


class MiniColumnV2(nn.Module):
    """
    Microcircuit canonique avec erreurs de prediction signees.

    Les deux populations L2/3 representent explicitement :
      - Ls_pos : bottom-up > top-down
      - Ls_neg : top-down > bottom-up
    """

    def __init__(
        self,
        dt: float = 1.0,
        r_max: float = R_MAX,
        theta_ca: float = 0.10,
        lambda_burst: float = 2.0,
    ):
        super().__init__()
        self.dt = dt
        self.r_max = r_max
        self.theta_ca = theta_ca
        self.lambda_burst = lambda_burst

        self.J = nn.Parameter(build_biological_J())
        self.b_thal = nn.Parameter(torch.tensor([1.0, 0.2, 0.2, 0.0, 0.0, 0.6, 0.0, 0.0]))
        self.b_top = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 0.4, 0.5, 0.0, 0.2, 1.0]))

        self.g_b4 = nn.Parameter(torch.tensor(1.5))
        self.g_atop = nn.Parameter(torch.tensor(1.2))
        self.g_bpv = nn.Parameter(torch.tensor(0.8))
        self.g_asst = nn.Parameter(torch.tensor(0.6))

        self.register_buffer("tau", TAU.clone())

        self.stp_pv_to_pos = STPSynapse(mode="STD", tau_d=200.0, tau_f=150.0, U=0.5, dt=dt)
        self.stp_pv_to_neg = STPSynapse(mode="STD", tau_d=200.0, tau_f=150.0, U=0.5, dt=dt)
        self.stp_sst_to_pos = STPSynapse(mode="STF", tau_d=300.0, tau_f=500.0, U=0.1, dt=dt)
        self.stp_sst_to_neg = STPSynapse(mode="STF", tau_d=300.0, tau_f=500.0, U=0.1, dt=dt)
        self.stp_vip_to_sst = STPSynapse(mode="STF", tau_d=200.0, tau_f=300.0, U=0.05, dt=dt)

    def reset_dynamic_state(self) -> None:
        self.stp_pv_to_pos.reset()
        self.stp_pv_to_neg.reset()
        self.stp_sst_to_pos.reset()
        self.stp_sst_to_neg.reset()
        self.stp_vip_to_sst.reset()

    def phi(self, x: torch.Tensor, beta: float = 0.10) -> torch.Tensor:
        return self.r_max / (1.0 + torch.exp(-beta * x))

    def _effective_connectivity(self, state: torch.Tensor) -> torch.Tensor:
        J_eff = self.J.clone()
        r_pv = state[..., IDX["PV"]].mean()
        r_sst = state[..., IDX["SST"]].mean()
        r_vip = state[..., IDX["VIP"]].mean()
        J_eff[IDX["Ls_pos"], IDX["PV"]] = self.stp_pv_to_pos.effective_weight(
            self.J[IDX["Ls_pos"], IDX["PV"]], r_pv
        )
        J_eff[IDX["Ls_neg"], IDX["PV"]] = self.stp_pv_to_neg.effective_weight(
            self.J[IDX["Ls_neg"], IDX["PV"]], r_pv
        )
        J_eff[IDX["Ls_pos"], IDX["SST"]] = self.stp_sst_to_pos.effective_weight(
            self.J[IDX["Ls_pos"], IDX["SST"]], r_sst
        )
        J_eff[IDX["Ls_neg"], IDX["SST"]] = self.stp_sst_to_neg.effective_weight(
            self.J[IDX["Ls_neg"], IDX["SST"]], r_sst
        )
        J_eff[IDX["SST"], IDX["VIP"]] = self.stp_vip_to_sst.effective_weight(
            self.J[IDX["SST"], IDX["VIP"]], r_vip
        )
        return J_eff

    def dendritic_step(self, state: torch.Tensor, s: torch.Tensor, mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        r_l4 = state[..., IDX["L4"]]
        r_pv = state[..., IDX["PV"]]
        r_sst = state[..., IDX["SST"]]

        basal = self.g_b4 * r_l4 + 0.25 * s - self.g_bpv * r_pv
        apical = self.g_atop * mu - self.g_asst * r_sst

        eps_pos = torch.relu((basal - apical) / self.r_max)
        eps_neg = torch.relu((apical - basal) / self.r_max)
        surprise = eps_pos + eps_neg
        p_ca = (surprise > self.theta_ca).float()

        ls_pos = self.phi(basal + self.lambda_burst * p_ca * eps_pos * self.r_max)
        ls_neg = self.phi(apical + self.lambda_burst * p_ca * eps_neg * self.r_max)

        return {
            "eps_pos": eps_pos,
            "eps_neg": eps_neg,
            "surprise": surprise,
            "p_ca": p_ca,
            "ls_pos": ls_pos,
            "ls_neg": ls_neg,
        }

    def forward(self, state: torch.Tensor, s: torch.Tensor, mu: torch.Tensor, u_ach: float = 0.0) -> MiniColumnOutput:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        s = torch.as_tensor(s, dtype=state.dtype, device=state.device).reshape(-1)
        mu = torch.as_tensor(mu, dtype=state.dtype, device=state.device).reshape(-1)
        if s.shape[0] == 1 and state.shape[0] > 1:
            s = s.expand(state.shape[0])
        if mu.shape[0] == 1 and state.shape[0] > 1:
            mu = mu.expand(state.shape[0])

        J_eff = self._effective_connectivity(state)
        recurrent = state @ J_eff.T
        thal_drive = s.unsqueeze(-1) * self.b_thal.unsqueeze(0)
        top_drive = mu.unsqueeze(-1) * self.b_top.unsqueeze(0)
        top_drive[:, IDX["VIP"]] += u_ach

        net = recurrent + thal_drive + top_drive
        drdt = (-state + self.phi(net)) / self.tau.to(state.device)
        new_state = torch.clamp(state + self.dt * drdt, 0.0, self.r_max)

        dendritic = self.dendritic_step(new_state, s, mu)
        new_state[:, IDX["Ls_pos"]] = dendritic["ls_pos"]
        new_state[:, IDX["Ls_neg"]] = dendritic["ls_neg"]

        return MiniColumnOutput(
            state=new_state,
            surprise=dendritic["surprise"],
            eps_pos=dendritic["eps_pos"],
            eps_neg=dendritic["eps_neg"],
            p_ca=dendritic["p_ca"],
        )

    @torch.no_grad()
    def plasticity_step(self, previous_state: torch.Tensor, output: MiniColumnOutput, eta: float = 1e-3, eta_dec: float = 1e-4) -> None:
        r_l4_prev = previous_state[..., IDX["L4"]].mean()
        r_vip = output.state[..., IDX["VIP"]].mean()
        gate = output.p_ca.mean() * (1.0 + 0.25 * r_vip / self.r_max)
        delta_pos = eta * output.state[..., IDX["Ls_pos"]].mean() * r_l4_prev * gate / self.r_max
        delta_neg = eta * output.state[..., IDX["Ls_neg"]].mean() * r_l4_prev * gate / self.r_max

        self.J.data[IDX["Ls_pos"], IDX["L4"]] += delta_pos
        self.J.data[IDX["Ls_neg"], IDX["L4"]] += 0.5 * delta_neg
        self.J.data[IDX["Ls_pos"], IDX["L4"]] -= eta_dec * self.J.data[IDX["Ls_pos"], IDX["L4"]]
        self.J.data[IDX["Ls_neg"], IDX["L4"]] -= eta_dec * self.J.data[IDX["Ls_neg"], IDX["L4"]]
        self.J.data[IDX["Ls_pos"], IDX["L4"]].clamp_(0.0)
        self.J.data[IDX["Ls_neg"], IDX["L4"]].clamp_(0.0)


class ColumnV2(nn.Module):
    """Colonne regroupant plusieurs minicolonnes avec competition k-WTA."""

    def __init__(self, n_mc: int = 256, k: int = 20, **mc_kwargs):
        super().__init__()
        self.n_mc = n_mc
        self.k = k
        self.minicolumns = nn.ModuleList([MiniColumnV2(**mc_kwargs) for _ in range(n_mc)])
        self.J_ff = nn.Parameter(torch.randn(n_mc, n_mc) * 0.01)
        self.J_fb = nn.Parameter(torch.randn(n_mc, n_mc) * 0.01)
        self.register_buffer("state", torch.zeros(n_mc, N_STATE))

    def reset_state(self) -> None:
        self.state.zero_()
        for mc in self.minicolumns:
            mc.reset_dynamic_state()

    def lateral_inhibition(self, surprise: torch.Tensor) -> torch.Tensor:
        gated = torch.zeros_like(surprise)
        top_idx = torch.topk(surprise, self.k).indices
        gated[top_idx] = surprise[top_idx]
        return gated

    def forward(self, s: torch.Tensor, mu: torch.Tensor | None = None, learn: bool = False) -> Dict[str, torch.Tensor]:
        s = torch.as_tensor(s, dtype=self.state.dtype, device=self.state.device)
        if s.ndim == 0:
            s = s.repeat(self.n_mc)
        if mu is None:
            mu = torch.zeros(self.n_mc, dtype=self.state.dtype, device=self.state.device)
        else:
            mu = torch.as_tensor(mu, dtype=self.state.dtype, device=self.state.device)
            if mu.ndim == 0:
                mu = mu.repeat(self.n_mc)

        outputs: List[MiniColumnOutput] = []
        new_states: List[torch.Tensor] = []
        previous_state = self.state.clone()
        for i, mc in enumerate(self.minicolumns):
            out = mc(self.state[i], s[i], mu[i])
            outputs.append(out)
            new_states.append(out.state.squeeze(0))
            if learn:
                mc.plasticity_step(previous_state[i].unsqueeze(0), out)

        new_state = torch.stack(new_states)
        surprise = torch.stack([out.surprise.squeeze(0) for out in outputs])
        eps_pos = torch.stack([out.eps_pos.squeeze(0) for out in outputs])
        eps_neg = torch.stack([out.eps_neg.squeeze(0) for out in outputs])
        p_ca = torch.stack([out.p_ca.squeeze(0) for out in outputs])

        sparse_surprise = self.lateral_inhibition(surprise)
        sdr = sparse_surprise > 0
        y_ff = sparse_surprise @ self.J_ff
        y_fb = new_state[:, IDX["Ld"]] @ self.J_fb

        self.state.copy_(new_state)

        return {
            "state": new_state,
            "sdr": sdr,
            "surprise": sparse_surprise,
            "eps_pos": eps_pos,
            "eps_neg": eps_neg,
            "p_ca": p_ca,
            "y_ff": y_ff,
            "y_fb": y_fb,
            "r_6": new_state[:, IDX["L6"]],
        }

    def thalamocortical_error(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=self.state.dtype, device=self.state.device)
        if s.ndim == 0:
            s = s.repeat(self.n_mc)
        mu_pred = self.state[:, IDX["L6"]]
        return s - mu_pred


class ColumnEnsembleV2(nn.Module):
    """Ensemble simple de colonnes organisees en hetararchie legere."""

    def __init__(self, n_col: int = 4, n_mc: int = 256, k: int = 20, **mc_kwargs):
        super().__init__()
        self.n_col = n_col
        self.n_mc = n_mc
        self.columns = nn.ModuleList([ColumnV2(n_mc=n_mc, k=k, **mc_kwargs) for _ in range(n_col)])
        self.W_ff = nn.Parameter(torch.randn(n_col, n_mc, n_mc) * 0.01)
        self.W_fb = nn.Parameter(torch.randn(n_col, n_mc, n_mc) * 0.01)

    def reset_state(self) -> None:
        for col in self.columns:
            col.reset_state()

    def forward(self, s_list: List[torch.Tensor], mu_top: torch.Tensor | None = None, learn: bool = False) -> Dict[str, object]:
        outputs: List[Dict[str, torch.Tensor]] = []
        mus: List[torch.Tensor] = []

        if mu_top is None:
            mus.append(torch.zeros(self.n_mc))
        else:
            mus.append(torch.as_tensor(mu_top).float())

        for i, col in enumerate(self.columns):
            mu_i = mus[i] if i < len(mus) else mus[-1]
            out = col(s_list[i], mu=mu_i, learn=learn)
            outputs.append(out)
            if i + 1 < self.n_col:
                mus.append(out["y_fb"] @ self.W_fb[i])

        sdr_stack = torch.stack([out["sdr"].float() for out in outputs])
        consensus = sdr_stack.mean(dim=0) > 0.5
        free_energy = sum(
            0.5 * col.thalamocortical_error(torch.as_tensor(s_list[i]).float()).pow(2).mean()
            for i, col in enumerate(self.columns)
        )

        return {
            "consensus": consensus,
            "free_energy": free_energy,
            "outputs": outputs,
        }

