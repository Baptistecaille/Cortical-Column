"""
MODULE 5 — DisplacementAlgebra
Définitions 5.1–5.3, Théorèmes 5.1–5.2
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class DisplacementAlgebra(nn.Module):
    """
    Algèbre des cellules de déplacement sur le produit de tores.

    Déf. 5.1 : L = T^{d_1} × ... × T^{d_K} — produit de tores
    Déf. 5.2 : D_{A→B} = L_B ⊖ L_A = (L_B - L_A) mod ℤ^{d_total}

    Thm 5.1 : invariance sous translation globale δ
    Thm 5.2 : composition de déplacements D_{A→C} = D_{A→B} ⊕ D_{B→C}

    ⚠️ Toutes les opérations utilisent torch.remainder — jamais l'opérateur -.
    """

    def __init__(self, dims: List[int]):
        """
        Args:
            dims : liste [d_1, d_2, ..., d_K] des dimensions de chaque tore
        """
        super().__init__()
        self.dims = dims
        self.total_dim = sum(dims)

    def subtract(self, L_B: torch.Tensor, L_A: torch.Tensor) -> torch.Tensor:
        """
        Déf. 5.2 : D_{A→B} = (L_B - L_A) mod ℤ^{d_total}

        Invariant I5.1 : résultat ∈ [0, 1)
        Invariant I5.2 : invariant sous translation globale δ
        """
        return torch.remainder(L_B - L_A, 1.0)

    def add(self, L: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Addition dans le groupe (L, ⊕) : L ⊕ D = (L + D) mod 1

        Invariant I5.1 : résultat ∈ [0, 1)
        Invariant I5.3 : add(L_A, subtract(L_B, L_A)) ≈ L_B
        """
        return torch.remainder(L + D, 1.0)

    def compose_displacements(
        self, D_AB: torch.Tensor, D_BC: torch.Tensor
    ) -> torch.Tensor:
        """
        Thm 5.2(i) : D_{A→C} = D_{A→B} ⊕ D_{B→C} = (D_AB + D_BC) mod 1

        Invariant I5.4 : compose(D_AB, D_BC) == subtract(L_C, L_A)
        """
        return torch.remainder(D_AB + D_BC, 1.0)

    def encode_composition(
        self,
        sdr_parent: torch.BoolTensor,
        sdr_subobject: torch.BoolTensor,
        L_parent: torch.Tensor,
        L_subobject: torch.Tensor,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.Tensor]:
        """
        Déf. 5.3 : encode le triplet ⟨SDR_parent, SDR_subobject, D_subobject⟩

        Returns:
            (sdr_parent, sdr_subobject, D)  où D = L_subobject ⊖ L_parent
        """
        D = self.subtract(L_subobject, L_parent)
        return sdr_parent, sdr_subobject, D

    def is_zero(self, D: torch.Tensor, atol: float = 1e-5) -> bool:
        """Vérifie si un déplacement est nul (utile pour les tests I5.3)."""
        return torch.allclose(D, torch.zeros_like(D), atol=atol)
