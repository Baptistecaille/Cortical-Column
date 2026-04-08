"""
MODULE 2 — SpatialPooler
Définitions 2.1–2.6, Théorèmes 2.1–2.3
"""

import math
import torch
import torch.nn as nn


class SpatialPooler(nn.Module):
    """
    Encodeur spatial avec apprentissage hebbien local.

    Transforme une entrée binaire I ∈ {0,1}^N_in en un SDR C ∈ {0,1}^N_mc
    avec exactement k = ⌈s·N_mc⌉ bits actifs (Thm 2.3).

    Apprentissage : règle de plasticité hebbienne locale (Déf. 2.6).
    Aucun autograd — mises à jour manuelles uniquement.
    """

    def __init__(
        self,
        N_in: int,
        N_mc: int,
        s: float = 0.02,
        tau_conn: float = 0.5,
        beta: float = 1.0,
        T_w: int = 1000,
        delta_p_plus: float = 0.05,
        delta_p_minus: float = 0.03,
    ):
        """
        Args:
            N_in        : dimension de l'entrée
            N_mc        : nombre de mini-colonnes
            s           : ratio de parcimonie cible (ex. 0.02)
            tau_conn    : seuil de connexion synaptique (Déf. 2.2)
            beta        : facteur d'amplification du boost (Déf. 2.3)
            T_w         : fenêtre temporelle pour le duty cycle
            delta_p_plus  : incrément hebbien (Déf. 2.6)
            delta_p_minus : décrément hebbien (Déf. 2.6)
        """
        super().__init__()
        self.N_in = N_in
        self.N_mc = N_mc
        self.k = math.ceil(s * N_mc)      # nb de colonnes actives — Thm 2.3
        self.tau_conn = tau_conn
        self.beta = beta
        self.T_w = T_w
        self.delta_p_plus = delta_p_plus
        self.delta_p_minus = delta_p_minus

        # Déf. 2.2 — permanences synaptiques ∈ [0, 1], initialisées autour de tau_conn
        self.permanences = nn.Parameter(
            torch.clamp(torch.randn(N_mc, N_in) * 0.1 + tau_conn, 0.0, 1.0),
            requires_grad=False,  # mise à jour manuelle
        )

        # Déf. 2.3 — cycle moyen d'activation (EMA)
        self.register_buffer("duty_cycle", torch.full((N_mc,), s))
        # Approximation locale : moyenne des voisins ≈ moyenne globale
        self.register_buffer("neighbor_avg", torch.full((N_mc,), s))

    def forward(self, I: torch.BoolTensor) -> torch.BoolTensor:
        """
        Encode l'entrée I en SDR C avec exactement k bits actifs.

        Étape 1 : poids effectifs binaires W (Déf. 2.2)
        Étape 2 : facteur boost homéostatique (Déf. 2.3)
        Étape 3 : chevauchement effectif (Déf. 2.4)
        Étape 4 : k-WTA — sélection des top-k (Déf. 2.5)

        Invariant I2.1 : C.sum() == k exactement
        """
        # Étape 1 — Poids effectifs
        W = (self.permanences >= self.tau_conn)  # (N_mc, N_in) bool

        # Étape 2 — Boost homéostatique
        boost = torch.exp(self.beta * (self.neighbor_avg - self.duty_cycle))  # (N_mc,)

        # Étape 3 — Chevauchement pondéré
        raw_overlap = (W.float() * I.float()).sum(dim=1)  # (N_mc,)
        overlap_eff = boost * raw_overlap                  # (N_mc,)

        # Étape 4 — k-WTA (non différentiable — voir notes d'implémentation)
        _, top_indices = torch.topk(overlap_eff, self.k)
        C = torch.zeros(self.N_mc, dtype=torch.bool, device=I.device)
        C[top_indices] = True
        return C  # Thm 2.3 garantit C.sum() == k

    @torch.no_grad()
    def hebbian_update(self, I: torch.BoolTensor, C: torch.BoolTensor):
        """
        Déf. 2.6 — Plasticité hebbienne locale (sans autograd).

        Pour chaque mini-colonne active :
          - permanence += delta_p_plus  si l'entrée correspondante est active
          - permanence -= delta_p_minus si l'entrée correspondante est inactive

        Invariant I2.2 : permanences ∈ [0, 1] après mise à jour.
        """
        active = C.unsqueeze(1)       # (N_mc, 1)
        input_on = I.unsqueeze(0)     # (1, N_in)

        inc_mask = active & input_on   # actif ET entrée active
        dec_mask = active & ~input_on  # actif ET entrée inactive

        self.permanences.data[inc_mask] += self.delta_p_plus
        self.permanences.data[dec_mask] -= self.delta_p_minus
        self.permanences.data.clamp_(0.0, 1.0)  # Invariant I2.2

    @torch.no_grad()
    def update_duty_cycle(self, C: torch.BoolTensor):
        """
        Déf. 2.3 — Moyenne mobile exponentielle de l'activité.
        alpha = 1 / T_w

        Invariant I2.5 : duty_cycle.mean() → s sur T_w itérations.
        """
        alpha = 1.0 / self.T_w
        self.duty_cycle.mul_(1 - alpha).add_(C.float() * alpha)
        self.neighbor_avg.fill_(self.duty_cycle.mean())
