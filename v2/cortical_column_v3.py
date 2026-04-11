"""
Architecture v3 de colonne corticale.

Cette version etend la v2 avec plusieurs briques de la roadmap :
  - erreurs de prediction signees PE+/PE-
  - boucle thalamo-corticale explicite (TC, hT, NRT)
  - plasticite synaptique a court terme (STP)
  - localisation L6 par modules de cellules grilles
  - routage predictif gamma/beta avec PAC simplifiee
  - astrocytes et plasticite a 4 facteurs simplifiee
  - configuration prospective avant apprentissage
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


CORTICAL_V3 = [
    "L4",
    "Ls_pos",
    "Ls_neg",
    "Ld",
    "L6",
    "PVb",
    "PVc",
    "SSTm",
    "SSTnm",
    "VIP1",
    "VIP2",
    "NGF",
]
THALAMIC_V3 = ["TC", "hT", "NRT"]
ALL_POPS_V3 = CORTICAL_V3 + THALAMIC_V3
IDX_V3: Dict[str, int] = {name: i for i, name in enumerate(ALL_POPS_V3)}

N_STATE_V3 = len(ALL_POPS_V3)
N_CORTICAL_V3 = len(CORTICAL_V3)
R_MAX_V3 = 100.0
TAU_V3 = torch.tensor(
    [
        15.0,   # L4
        20.0,   # Ls_pos
        20.0,   # Ls_neg
        25.0,   # Ld
        40.0,   # L6
        5.0,    # PVb
        7.0,    # PVc
        30.0,   # SSTm
        20.0,   # SSTnm
        15.0,   # VIP1
        15.0,   # VIP2
        40.0,   # NGF
        20.0,   # TC
        100.0,  # hT
        15.0,   # NRT
    ]
)


def build_v3_connectivity() -> torch.Tensor:
    """Connectivite de base de la v3-beta etendue avec PE signes."""
    J = torch.zeros(N_STATE_V3, N_STATE_V3)
    i = IDX_V3

    # Excitation corticale
    J[i["L4"], i["TC"]] = 1.20
    J[i["L4"], i["L4"]] = 0.30
    J[i["Ls_pos"], i["L4"]] = 1.50
    J[i["Ls_neg"], i["L4"]] = 0.90
    J[i["Ld"], i["Ls_pos"]] = 0.90
    J[i["Ld"], i["Ls_neg"]] = 0.70
    J[i["L6"], i["Ls_pos"]] = 0.40
    J[i["L6"], i["Ls_neg"]] = 0.40
    J[i["L6"], i["Ld"]] = 0.30
    J[i["L6"], i["L6"]] = 0.50
    J[i["PVb"], i["L4"]] = 0.90
    J[i["PVb"], i["Ls_pos"]] = 0.50
    J[i["PVb"], i["Ls_neg"]] = 0.30
    J[i["PVc"], i["L4"]] = 0.70
    J[i["SSTm"], i["Ld"]] = 0.45
    J[i["SSTm"], i["Ls_pos"]] = 0.20
    J[i["SSTnm"], i["L4"]] = 0.50
    J[i["NGF"], i["Ls_pos"]] = 0.20
    J[i["NGF"], i["L6"]] = 0.10
    J[i["NRT"], i["TC"]] = 0.70

    # Inhibition
    J[i["Ls_pos"], i["PVb"]] = -0.80
    J[i["Ls_neg"], i["PVb"]] = -0.60
    J[i["Ls_pos"], i["PVc"]] = -0.50
    J[i["Ls_neg"], i["PVc"]] = -0.35
    J[i["Ls_pos"], i["SSTm"]] = -0.40
    J[i["Ls_neg"], i["SSTm"]] = -0.40
    J[i["Ls_pos"], i["NGF"]] = -0.20
    J[i["Ld"], i["PVb"]] = -0.50
    J[i["Ld"], i["SSTm"]] = -0.30
    J[i["L4"], i["SSTnm"]] = -0.40
    J[i["L4"], i["L6"]] = -0.30
    J[i["PVb"], i["VIP2"]] = -0.20
    J[i["SSTm"], i["VIP1"]] = -0.80
    J[i["SSTnm"], i["VIP1"]] = -0.30
    J[i["TC"], i["NRT"]] = -0.90
    J[i["TC"], i["L6"]] = -0.30
    J[i["L6"], i["NRT"]] = -0.20

    return J


class STPSynapseV3:
    """Tsodyks-Markram minimal, utilisable dans le forward sans autograd."""

    def __init__(
        self,
        mode: str = "STD",
        tau_d: float = 200.0,
        tau_f: float = 500.0,
        U: float = 0.5,
        dt: float = 0.1,
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
        r = torch.as_tensor(r_pre).float().mean() / R_MAX_V3
        dx = (1.0 - self.x) / self.tau_d - self.u * self.x * r
        if self.mode == "STF":
            du = (self.U - self.u) / self.tau_f + self.U * (1.0 - self.u) * r
        else:
            du = (self.U - self.u) / self.tau_f
        self.x = torch.clamp(self.x + self.dt * dx, 0.0, 1.0)
        self.u = torch.clamp(self.u + self.dt * du, 0.0, 1.5)

    def weight(self, base_weight: torch.Tensor, r_pre: torch.Tensor) -> torch.Tensor:
        self.update(r_pre)
        return base_weight * self.x * self.u


class GridCellLocationModule(nn.Module):
    """Code allocentrique multi-echelle pour L6."""

    def __init__(self, n_modules: int = 8, dt: float = 0.1):
        super().__init__()
        self.n_modules = n_modules
        self.dt = dt
        self.scales = nn.Parameter(torch.logspace(-1, 1, steps=n_modules))
        self.orientations = nn.Parameter(2 * math.pi * torch.rand(n_modules))
        self.anchor_gain = nn.Parameter(torch.tensor(0.15))
        self.register_buffer("phi", torch.zeros(n_modules, 2))

    @staticmethod
    def _rotation(theta: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta)
        s = torch.sin(theta)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    def reset(self) -> None:
        self.phi.zero_()

    def integrate(self, velocity: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        velocity = torch.as_tensor(velocity, dtype=self.phi.dtype, device=self.phi.device)
        if velocity.numel() < 2:
            velocity = torch.nn.functional.pad(velocity.flatten(), (0, 2 - velocity.numel()))
        velocity = velocity[:2]
        step_dt = self.dt if dt is None else dt
        for k in range(self.n_modules):
            rot = self._rotation(self.orientations[k])
            delta = (rot @ velocity) * step_dt / self.scales[k]
            self.phi[k] = torch.remainder(self.phi[k] + delta, 2 * math.pi)
        return self.get_code()

    def anchor(self, sensory_signal: torch.Tensor) -> torch.Tensor:
        signal = torch.as_tensor(sensory_signal, dtype=self.phi.dtype, device=self.phi.device).flatten()
        if signal.numel() == 0:
            return self.get_code()
        anchor = signal.mean().tanh()
        self.phi = torch.remainder(self.phi + self.anchor_gain * anchor, 2 * math.pi)
        return self.get_code()

    def get_code(self) -> torch.Tensor:
        return torch.cat([torch.cos(self.phi[:, 0]), torch.sin(self.phi[:, 1])], dim=0)


class AstrocyteLayer(nn.Module):
    """Variable lente de contexte et de consolidation."""

    def __init__(self, tau_astro: float = 5000.0, dt: float = 0.1):
        super().__init__()
        self.tau_astro = tau_astro
        self.dt = dt
        self.theta_ca = nn.Parameter(torch.tensor(0.30))
        self.g = nn.Parameter(torch.tensor(0.50))
        self.register_buffer("ca", torch.zeros(1))

    def reset(self) -> None:
        self.ca.zero_()

    def step(self, synaptic_activity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activity = torch.as_tensor(synaptic_activity, dtype=self.ca.dtype, device=self.ca.device).mean()
        dca = (-self.ca / self.tau_astro) + 0.01 * activity
        self.ca = torch.clamp(self.ca + self.dt * dca, 0.0, 1.0)
        glio = torch.sigmoid(self.ca - self.theta_ca)
        weight_mod = 1.0 + self.g * glio
        plasticity_gate = torch.sigmoid(-self.ca)
        return weight_mod.squeeze(0), plasticity_gate.squeeze(0)


@dataclass
class MiniColumnV3Output:
    state: torch.Tensor
    eps_pos: torch.Tensor
    eps_neg: torch.Tensor
    p_ca_soft: torch.Tensor
    p_ca_hard: torch.Tensor
    eps_th: torch.Tensor
    glio_gain: torch.Tensor
    astro_gate: torch.Tensor


class MiniColumnV3(nn.Module):
    """Minicolonne v3 avec thalamus, PE signes, STP et astrocyte."""

    W_TC6 = 0.30
    W_TC_NRT = 0.70
    W_NRT_TC = 0.90
    W_NRT_6 = 0.80
    G_T = 0.50

    def __init__(
        self,
        dt: float = 0.1,
        theta_ca: float = 0.15,
        lambda_burst: float = 2.5,
        n_grid_modules: int = 8,
    ):
        super().__init__()
        self.dt = dt
        self.theta_ca = theta_ca
        self.lambda_burst = lambda_burst

        self.J = nn.Parameter(build_v3_connectivity())
        self.b_sensory = nn.Parameter(torch.zeros(N_STATE_V3))
        self.b_top = nn.Parameter(torch.zeros(N_STATE_V3))
        self.b_sensory.data[IDX_V3["TC"]] = 1.0
        self.b_sensory.data[IDX_V3["L4"]] = 0.3
        self.b_top.data[IDX_V3["VIP1"]] = 0.4
        self.b_top.data[IDX_V3["Ls_neg"]] = 0.6
        self.b_top.data[IDX_V3["Ld"]] = 0.5
        self.b_top.data[IDX_V3["L6"]] = 0.7

        self.g_b4 = nn.Parameter(torch.tensor(1.5))
        self.g_atop = nn.Parameter(torch.tensor(1.2))
        self.g_bpv = nn.Parameter(torch.tensor(0.8))
        self.g_asst = nn.Parameter(torch.tensor(0.6))
        self.g_c = nn.Parameter(torch.tensor(0.1))

        self.theta_ltp = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("tau", TAU_V3.clone())

        self.grid = GridCellLocationModule(n_modules=n_grid_modules, dt=dt)
        self.astro = AstrocyteLayer(dt=dt)

        self.stp_pv_to_pos = STPSynapseV3(mode="STD", tau_d=200.0, tau_f=150.0, U=0.5, dt=dt)
        self.stp_pv_to_neg = STPSynapseV3(mode="STD", tau_d=200.0, tau_f=150.0, U=0.5, dt=dt)
        self.stp_sst_to_pos = STPSynapseV3(mode="STF", tau_d=300.0, tau_f=500.0, U=0.1, dt=dt)
        self.stp_sst_to_neg = STPSynapseV3(mode="STF", tau_d=300.0, tau_f=500.0, U=0.1, dt=dt)
        self.stp_vip_to_sst = STPSynapseV3(mode="STF", tau_d=200.0, tau_f=300.0, U=0.05, dt=dt)

    def reset_dynamic_state(self) -> None:
        self.grid.reset()
        self.astro.reset()
        self.stp_pv_to_pos.reset()
        self.stp_pv_to_neg.reset()
        self.stp_sst_to_pos.reset()
        self.stp_sst_to_neg.reset()
        self.stp_vip_to_sst.reset()

    @staticmethod
    def phi(x: torch.Tensor, beta: float = 0.10) -> torch.Tensor:
        return R_MAX_V3 / (1.0 + torch.exp(-beta * x))

    @staticmethod
    def h_inf(r_tc: torch.Tensor) -> torch.Tensor:
        v_proxy = -70.0 + 0.5 * r_tc
        return 1.0 / (1.0 + torch.exp((v_proxy + 65.0) / 5.0))

    def _effective_connectivity(self, state: torch.Tensor) -> torch.Tensor:
        J_eff = self.J.clone()
        r_pvb = state[..., IDX_V3["PVb"]].mean()
        r_sstm = state[..., IDX_V3["SSTm"]].mean()
        r_vip1 = state[..., IDX_V3["VIP1"]].mean()
        J_eff[IDX_V3["Ls_pos"], IDX_V3["PVb"]] = self.stp_pv_to_pos.weight(
            self.J[IDX_V3["Ls_pos"], IDX_V3["PVb"]], r_pvb
        )
        J_eff[IDX_V3["Ls_neg"], IDX_V3["PVb"]] = self.stp_pv_to_neg.weight(
            self.J[IDX_V3["Ls_neg"], IDX_V3["PVb"]], r_pvb
        )
        J_eff[IDX_V3["Ls_pos"], IDX_V3["SSTm"]] = self.stp_sst_to_pos.weight(
            self.J[IDX_V3["Ls_pos"], IDX_V3["SSTm"]], r_sstm
        )
        J_eff[IDX_V3["Ls_neg"], IDX_V3["SSTm"]] = self.stp_sst_to_neg.weight(
            self.J[IDX_V3["Ls_neg"], IDX_V3["SSTm"]], r_sstm
        )
        J_eff[IDX_V3["SSTm"], IDX_V3["VIP1"]] = self.stp_vip_to_sst.weight(
            self.J[IDX_V3["SSTm"], IDX_V3["VIP1"]], r_vip1
        )
        return J_eff

    def dendritic_step(
        self,
        state: torch.Tensor,
        s: torch.Tensor,
        mu: torch.Tensor,
        grid_code: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r_l4 = state[..., IDX_V3["L4"]]
        r_pvb = state[..., IDX_V3["PVb"]]
        r_sstm = state[..., IDX_V3["SSTm"]]
        grid_signal = 0.15 * grid_code.mean()

        basal = self.g_b4 * r_l4 + 0.50 * s - self.g_bpv * r_pvb + grid_signal
        apical = self.g_atop * mu - self.g_asst * r_sstm + self.g_c * state[..., IDX_V3["L6"]]

        eps_pos = torch.relu((basal - apical) / R_MAX_V3)
        eps_neg = torch.relu((apical - basal) / R_MAX_V3)
        mismatch = eps_pos + eps_neg
        p_ca_soft = torch.tanh(mismatch / self.theta_ca)
        p_ca_hard = (mismatch > self.theta_ca).float()

        ls_pos = self.phi(basal + self.lambda_burst * p_ca_soft * eps_pos * R_MAX_V3)
        ls_neg = self.phi(apical + self.lambda_burst * p_ca_soft * eps_neg * R_MAX_V3)
        return {
            "eps_pos": eps_pos,
            "eps_neg": eps_neg,
            "p_ca_soft": p_ca_soft,
            "p_ca_hard": p_ca_hard,
            "ls_pos": ls_pos,
            "ls_neg": ls_neg,
        }

    def forward(
        self,
        state: torch.Tensor,
        s: torch.Tensor,
        mu: torch.Tensor,
        velocity: torch.Tensor | None = None,
        u_ach: float = 0.0,
        anchor: bool = True,
    ) -> MiniColumnV3Output:
        if state.ndim == 1:
            state = state.unsqueeze(0)

        s = torch.as_tensor(s, dtype=state.dtype, device=state.device).reshape(-1)
        mu = torch.as_tensor(mu, dtype=state.dtype, device=state.device).reshape(-1)
        if s.shape[0] == 1 and state.shape[0] > 1:
            s = s.expand(state.shape[0])
        if mu.shape[0] == 1 and state.shape[0] > 1:
            mu = mu.expand(state.shape[0])

        if velocity is None:
            velocity = torch.zeros(2, dtype=state.dtype, device=state.device)
        grid_code = self.grid.integrate(velocity, dt=self.dt)
        if anchor:
            grid_code = self.grid.anchor(s)

        J_eff = self._effective_connectivity(state)
        recurrent = state @ J_eff.T
        sensory_drive = s.unsqueeze(-1) * self.b_sensory.unsqueeze(0)
        top_drive = mu.unsqueeze(-1) * self.b_top.unsqueeze(0)
        top_drive[:, IDX_V3["VIP1"]] += u_ach
        total_drive = recurrent + sensory_drive + top_drive

        glio_gain, astro_gate = self.astro.step(state[..., :N_CORTICAL_V3].mean())
        total_drive[..., :N_CORTICAL_V3] = total_drive[..., :N_CORTICAL_V3] * glio_gain

        drdt = (-state + self.phi(total_drive)) / self.tau.to(state.device)
        new_state = torch.clamp(state + self.dt * drdt, 0.0, R_MAX_V3)

        # Boucle thalamique explicite
        r_tc = new_state[..., IDX_V3["TC"]]
        h_t = state[..., IDX_V3["hT"]]
        r_nrt = new_state[..., IDX_V3["NRT"]]
        r_l6 = new_state[..., IDX_V3["L6"]]
        eps_th = s - (0.5 * r_l6 + 0.15 * grid_code.mean())
        h_t_inf = self.h_inf(r_tc)
        new_state[..., IDX_V3["hT"]] = torch.clamp(h_t + self.dt * (h_t_inf - h_t) / self.tau[IDX_V3["hT"]], 0.0, 1.0)
        burst = self.G_T * new_state[..., IDX_V3["hT"]] * torch.relu(-eps_th / R_MAX_V3)
        new_state[..., IDX_V3["TC"]] = torch.clamp(
            r_tc + self.dt * (eps_th + burst - self.W_NRT_TC * r_nrt) / self.tau[IDX_V3["TC"]],
            0.0,
            R_MAX_V3,
        )
        new_state[..., IDX_V3["NRT"]] = torch.clamp(
            r_nrt + self.dt * (self.W_TC_NRT * new_state[..., IDX_V3["TC"]] + self.W_NRT_6 * r_l6 - r_nrt)
            / self.tau[IDX_V3["NRT"]],
            0.0,
            R_MAX_V3,
        )

        dendritic = self.dendritic_step(new_state, s, mu, grid_code)
        new_state[..., IDX_V3["Ls_pos"]] = dendritic["ls_pos"]
        new_state[..., IDX_V3["Ls_neg"]] = dendritic["ls_neg"]
        new_state[..., IDX_V3["L6"]] = torch.clamp(
            new_state[..., IDX_V3["L6"]] + 0.1 * grid_code.mean(),
            0.0,
            R_MAX_V3,
        )

        return MiniColumnV3Output(
            state=new_state,
            eps_pos=dendritic["eps_pos"],
            eps_neg=dendritic["eps_neg"],
            p_ca_soft=dendritic["p_ca_soft"],
            p_ca_hard=dendritic["p_ca_hard"],
            eps_th=eps_th,
            glio_gain=glio_gain,
            astro_gate=astro_gate,
        )

    @torch.no_grad()
    def plasticity_step(self, prev_state: torch.Tensor, output: MiniColumnV3Output, neuromod: float = 1.0, eta: float = 5e-4) -> None:
        r_l4_prev = prev_state[..., IDX_V3["L4"]].mean()
        r_vip1 = output.state[..., IDX_V3["VIP1"]].mean()
        gate_meta = torch.sigmoid(self.theta_ltp - self.J[IDX_V3["Ls_pos"], IDX_V3["L4"]])
        eligibility = output.state[..., IDX_V3["Ls_pos"]].mean() * r_l4_prev / R_MAX_V3
        delta = eta * eligibility * output.p_ca_soft.mean() * neuromod * output.astro_gate * gate_meta
        self.J.data[IDX_V3["Ls_pos"], IDX_V3["L4"]] += delta * (1.0 + 0.1 * r_vip1 / R_MAX_V3)
        self.J.data[IDX_V3["Ls_neg"], IDX_V3["L4"]] += 0.5 * delta
        self.J.data[IDX_V3["Ls_pos"], IDX_V3["L4"]].clamp_(0.0, 3.0)
        self.J.data[IDX_V3["Ls_neg"], IDX_V3["L4"]].clamp_(0.0, 3.0)
        self.theta_ltp.data += 1e-4 * (self.J.data[IDX_V3["Ls_pos"], IDX_V3["L4"]] - self.theta_ltp.data)

    def prospective_step(
        self,
        state: torch.Tensor,
        s: torch.Tensor,
        mu: torch.Tensor,
        velocity: torch.Tensor | None = None,
        n_inf: int = 5,
        learn: bool = False,
    ) -> MiniColumnV3Output:
        current = state
        out: MiniColumnV3Output | None = None
        for _ in range(n_inf):
            out = self(current, s=s, mu=mu, velocity=velocity)
            current = out.state.detach()
        if learn and out is not None:
            self.plasticity_step(state, out)
        return out


class ColumnV3(nn.Module):
    """Colonne v3 avec routage predictif et memoire de pattern."""

    def __init__(self, n_mc: int = 128, k: int = 16, dt: float = 0.1, n_grid_modules: int = 8):
        super().__init__()
        self.n_mc = n_mc
        self.k = k
        self.minicolumns = nn.ModuleList(
            [MiniColumnV3(dt=dt, n_grid_modules=n_grid_modules) for _ in range(n_mc)]
        )
        self.J_ff = nn.Parameter(torch.randn(n_mc, n_mc) * 0.01)
        self.J_fb = nn.Parameter(torch.randn(n_mc, n_mc) * 0.01)
        self.pattern_memory = nn.Parameter(torch.zeros(n_mc))
        self.register_buffer("state", torch.zeros(n_mc, N_STATE_V3))

    def reset_state(self) -> None:
        self.state.zero_()
        for mc in self.minicolumns:
            mc.reset_dynamic_state()

    def lateral_inhibition(self, drive: torch.Tensor) -> torch.Tensor:
        sparse = torch.zeros_like(drive)
        top_idx = torch.topk(drive, self.k).indices
        sparse[top_idx] = drive[top_idx]
        return sparse

    def forward(
        self,
        s: torch.Tensor,
        mu: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        learn: bool = False,
        prospective_steps: int = 3,
    ) -> Dict[str, torch.Tensor]:
        s = torch.as_tensor(s, dtype=self.state.dtype, device=self.state.device)
        if s.ndim == 0:
            s = s.repeat(self.n_mc)
        if mu is None:
            mu = torch.zeros(self.n_mc, dtype=self.state.dtype, device=self.state.device)
        else:
            mu = torch.as_tensor(mu, dtype=self.state.dtype, device=self.state.device)
            if mu.ndim == 0:
                mu = mu.repeat(self.n_mc)
        if velocity is None:
            velocity = torch.zeros(2, dtype=self.state.dtype, device=self.state.device)
        else:
            velocity = torch.as_tensor(velocity, dtype=self.state.dtype, device=self.state.device)

        prev_state = self.state.clone()
        outputs: List[MiniColumnV3Output] = []
        states: List[torch.Tensor] = []
        for idx, mc in enumerate(self.minicolumns):
            out = mc.prospective_step(
                prev_state[idx],
                s=s[idx],
                mu=mu[idx],
                velocity=velocity,
                n_inf=prospective_steps,
                learn=learn,
            )
            outputs.append(out)
            states.append(out.state.squeeze(0))

        new_state = torch.stack(states)
        eps_pos = torch.stack([o.eps_pos.squeeze(0) for o in outputs])
        eps_neg = torch.stack([o.eps_neg.squeeze(0) for o in outputs])
        p_ca_soft = torch.stack([o.p_ca_soft.squeeze(0) for o in outputs])
        p_ca_hard = torch.stack([o.p_ca_hard.squeeze(0) for o in outputs])
        eps_th = torch.stack([o.eps_th.squeeze(0) for o in outputs])
        astro_gate = torch.stack([o.astro_gate for o in outputs])

        r_l6 = new_state[:, IDX_V3["L6"]]
        prediction_strength = torch.sigmoid(r_l6 * self.pattern_memory)
        gamma_drive = new_state[:, IDX_V3["Ls_pos"]] + new_state[:, IDX_V3["Ls_neg"]]
        sparse_gamma = self.lateral_inhibition(gamma_drive)
        gamma_gate = 1.0 - prediction_strength

        y_ff = (sparse_gamma * gamma_gate) @ self.J_ff
        y_fb = new_state[:, IDX_V3["Ld"]] @ self.J_fb

        # PAC simplifiee
        beta_phase = torch.angle(torch.fft.fft(new_state[:, IDX_V3["Ld"]].to(torch.complex64)))[1]
        pac_factor = 1.0 + 0.2 * torch.cos(beta_phase)
        y_ff_pac = y_ff * pac_factor.real

        if learn:
            self.pattern_memory.data = 0.99 * self.pattern_memory.data + 0.01 * r_l6.detach()

        self.state.copy_(new_state)

        return {
            "state": new_state,
            "sdr": sparse_gamma > 0,
            "y_ff": y_ff_pac,
            "y_fb": y_fb,
            "gamma_gate": gamma_gate,
            "prediction_strength": prediction_strength,
            "eps_pos": eps_pos,
            "eps_neg": eps_neg,
            "p_ca_soft": p_ca_soft,
            "p_ca_hard": p_ca_hard,
            "eps_th": eps_th,
            "astro_gate": astro_gate,
            "r_6": r_l6,
        }

    def thalamocortical_error(self, s: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(s, dtype=self.state.dtype, device=self.state.device)
        if s.ndim == 0:
            s = s.repeat(self.n_mc)
        return s - self.state[:, IDX_V3["L6"]]


class ColumnEnsembleV3(nn.Module):
    """Ensemble de colonnes v3 avec routage FF/FB simple."""

    def __init__(self, n_col: int = 4, n_mc: int = 128, k: int = 16, dt: float = 0.1, n_grid_modules: int = 8):
        super().__init__()
        self.n_col = n_col
        self.n_mc = n_mc
        self.columns = nn.ModuleList(
            [ColumnV3(n_mc=n_mc, k=k, dt=dt, n_grid_modules=n_grid_modules) for _ in range(n_col)]
        )
        self.W_fb = nn.Parameter(torch.randn(n_col, n_mc, n_mc) * 0.01)

    def reset_state(self) -> None:
        for col in self.columns:
            col.reset_state()

    def forward(
        self,
        s_list: List[torch.Tensor],
        velocity_list: List[torch.Tensor] | None = None,
        mu_top: torch.Tensor | None = None,
        learn: bool = False,
    ) -> Dict[str, object]:
        if velocity_list is None:
            velocity_list = [torch.zeros(2) for _ in range(self.n_col)]

        outputs: List[Dict[str, torch.Tensor]] = []
        mus: List[torch.Tensor] = []
        mus.append(torch.zeros(self.n_mc) if mu_top is None else torch.as_tensor(mu_top).float())

        for i, col in enumerate(self.columns):
            mu_i = mus[i] if i < len(mus) else mus[-1]
            out = col(s_list[i], mu=mu_i, velocity=velocity_list[i], learn=learn)
            outputs.append(out)
            if i + 1 < self.n_col:
                mus.append(out["y_fb"] @ self.W_fb[i])

        sdr_stack = torch.stack([out["sdr"].float() for out in outputs])
        gamma_stack = torch.stack([out["gamma_gate"] for out in outputs])
        consensus = (sdr_stack * gamma_stack).mean(dim=0) > 0.35
        free_energy = sum(
            0.5 * col.thalamocortical_error(torch.as_tensor(s_list[i]).float()).pow(2).mean()
            for i, col in enumerate(self.columns)
        )

        return {
            "consensus": consensus,
            "free_energy": free_energy,
            "outputs": outputs,
        }
