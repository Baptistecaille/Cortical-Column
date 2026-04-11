"""
MODULE 3 — Layer6bTransformer
Définitions 4.1–4.3, Théorèmes 4.1–4.2
"""

import math
import torch
import torch.nn as nn


class Layer6bTransformer(nn.Module):
    """
    Transformation thalamique égocentrique → allocentrique.

    Convertit une vitesse v_ego (référentiel capteur) en v_allo (référentiel objet)
    via une rotation R(θ) ∈ SO(d).

    L'orientation est maintenue dans un tampon circulaire discret (Déf. 4.3).
    Propriété clé : invariance allocentrique sous rotation du capteur (Thm 4.1).
    """

    def __init__(self, d: int = 2, N_orientations: int = 360):
        """
        Args:
            d               : dimension de l'espace de mouvement (2 ou 3)
            N_orientations  : résolution du tampon circulaire (Déf. 4.3)
        """
        super().__init__()
        assert d in (2, 3), f"Seul d=2 ou d=3 est supporté, reçu d={d}"
        self.d = d
        self.N = N_orientations

        # Déf. 4.3 : index courant dans le tampon discret ∈ [0, N-1]
        # Thm 4.2 : forme un groupe abélien (ℤ/Nℤ, +)
        self.register_buffer("theta_idx", torch.tensor(0, dtype=torch.long))

    def _rotation_matrix(self, theta_rad: float, device) -> torch.Tensor:
        """
        Déf. 4.2 : R(θ) ∈ SO(2) pour d=2.
        Invariant I4.1 : R^T R = I et det(R) = 1.
        """
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=device)
        return R

    def _rotation_matrix_3d(self, theta_rad: float, device) -> torch.Tensor:
        """
        Extension SO(3) pour d=3 : rotation autour de l'axe Z.
        Invariant I4.1 : R^T R = I et det(R) = 1.
        """
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        R = torch.tensor(
            [[c, -s, 0.0],
             [s,  c, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        return R

    def update_orientation(self, delta_n: int):
        """
        Déf. 4.3 : r_{i,n} = (i + n) mod N

        Invariant I4.3 : theta_idx ∈ [0, N-1] après toute mise à jour.
        Invariant I4.5 : update_orientation(N) → theta_idx inchangé.
        """
        self.theta_idx = (self.theta_idx + delta_n) % self.N

    def transform(self, v_ego: torch.Tensor) -> torch.Tensor:
        """
        Déf. 4.2 : v_allo = R(θ_{L6b}) · v_ego

        Invariant I4.2 : ||v_allo|| == ||v_ego||  (rotation conserve la norme)
        Invariant I4.4 : invariance allocentrique — même angle total → même résultat.
        """
        theta_rad = (self.theta_idx.item() / self.N) * 2 * math.pi

        if self.d == 2:
            R = self._rotation_matrix(theta_rad, v_ego.device)
        else:
            R = self._rotation_matrix_3d(theta_rad, v_ego.device)

        return R @ v_ego

    def get_rotation_matrix(self) -> torch.Tensor:
        """Retourne la matrice de rotation courante (utile pour les tests I4.1)."""
        theta_rad = (self.theta_idx.item() / self.N) * 2 * math.pi
        if self.d == 2:
            return self._rotation_matrix(theta_rad, device=torch.device("cpu"))
        return self._rotation_matrix_3d(theta_rad, device=torch.device("cpu"))
