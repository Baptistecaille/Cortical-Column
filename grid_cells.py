"""
MODULE 4 — GridCellNetwork
Définitions 3.1–3.4, Théorèmes 3.1–3.2
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional


def _first_k_primes(k: int) -> List[int]:
    """Retourne les k premiers nombres premiers (pour Thm 3.2 — CRT)."""
    primes = []
    candidate = 2
    while len(primes) < k:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


class GridCellModule(nn.Module):
    """
    Module individuel de cellules de grille — intégrateur de chemin sur tore.

    Déf. 3.1–3.2 : état interne = phase φ ∈ [0, 1)^d sur un tore de période λ.
    Déf. 3.2 : dφ/dt = A·v(t), intégration discrète φ ← (φ + A·v·dt) mod 1
    Thm 3.1 : φ(T) = φ_0 + A·∫v dt  (path-independent, dépend du déplacement total)
    """

    def __init__(self, d: int = 2, spatial_period: float = 1.0):
        super().__init__()
        self.d = d
        self.spatial_period = spatial_period  # Déf. 3.4 : λ

        # Déf. 3.2 : matrice de projection A ∈ ℝ^{d×d}
        self.A = nn.Parameter(
            torch.eye(d, dtype=torch.float32) * (1.0 / spatial_period)
        )
        # État courant sur le tore — Invariant I3.1 : phase ∈ [0, 1)^d
        self.register_buffer("phase", torch.zeros(d, dtype=torch.float32))

    def integrate(self, v: torch.Tensor, dt: float = 1.0):
        """
        Déf. 3.2 : φ ← (φ + A·v·dt) mod ℤ^d

        Invariant I3.1 : phase ∈ [0, 1) après chaque intégration.
        ⚠️ Utiliser torch.remainder (gère les négatifs), jamais l'opérateur %.
        """
        delta_phi = (self.A @ v) * dt
        self.phase = torch.remainder(self.phase + delta_phi, 1.0)

    def reset(self, phase: Optional[torch.Tensor] = None):
        """Réinitialise la phase (utile pour les tests de path-independence)."""
        if phase is None:
            self.phase.zero_()
        else:
            self.phase.copy_(torch.remainder(phase, 1.0))


class GridCellNetwork(nn.Module):
    """
    Réseau de K modules de cellules de grille avec périodes premières entre elles.

    Thm 3.2 (CRT) : K modules à périodes premières entre elles distinguent
    Λ = ∏ λᵢ positions distinctes dans l'espace.

    État : L_t = concaténation des phases de tous les modules → (K*d,) float ∈ [0,1)
    """

    def __init__(self, K: int = 4, d: int = 2, periods: Optional[List[float]] = None):
        """
        Args:
            K       : nombre de modules
            d       : dimension de chaque tore
            periods : périodes spatiales — si None, utilise les K premiers nombres premiers
        """
        super().__init__()
        if periods is None:
            periods = [float(p) for p in _first_k_primes(K)]

        assert len(periods) == K, "len(periods) doit être égal à K"

        self.K = K
        self.d = d
        self.periods = periods
        # Capacité totale (Thm 3.2) — produit des périodes si premières entre elles
        self.Lambda = math.prod(int(p) for p in periods)

        self.grid_modules = nn.ModuleList(
            [GridCellModule(d, lam) for lam in periods]
        )

    def integrate_all(self, v_allo: torch.Tensor, dt: float = 1.0):
        """
        Déf. 3.3 : L_{t+Δt} = T(L_t, v_allo)
        Intègre tous les modules avec la même vitesse allocentrique.
        """
        for mod in self.grid_modules:
            mod.integrate(v_allo, dt)

    def get_location_state(self) -> torch.Tensor:
        """
        Retourne L_t = concaténation des phases de tous les modules.

        Invariant I3.1 : résultat ∈ [0, 1)^{K*d}
        """
        return torch.cat([mod.phase for mod in self.grid_modules])  # (K*d,)

    def reset_all(self, phase: Optional[torch.Tensor] = None):
        """Réinitialise tous les modules (utile pour les tests de path-independence)."""
        for mod in self.grid_modules:
            mod.reset(phase)
