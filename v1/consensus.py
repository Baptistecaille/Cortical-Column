"""
MODULE 6 — MultiColumnConsensus
Définitions 6.1–6.2, Théorèmes 6.1–6.3
"""

import torch
import torch.nn as nn
from typing import List


class MultiColumnConsensus(nn.Module):
    """
    Vote distal et consensus entre K colonnes corticales.

    Déf. 6.1 : union locale U^(c) = OR booléen de toutes les hypothèses d'une colonne
    Déf. 6.2 : consensus C = AND booléen des unions de toutes les colonnes

    Thm 6.1 : supp(C) = ∩ supp(U^(c))
    Thm 6.3 : P_fp(K) ≤ ρ^{Kw} — décroissance exponentielle avec K
    Cor. 6.4 : réduction parallèle en arbre → O(log K) étapes
    """

    def __init__(self, n: int, K: int):
        """
        Args:
            n : dimension des SDR
            K : nombre de colonnes
        """
        super().__init__()
        self.n = n
        self.K = K

    def compute_union(self, hypotheses: List[torch.BoolTensor]) -> torch.BoolTensor:
        """
        Déf. 6.1 : U^(c) = OR booléen de toutes les hypothèses locales.

        Invariant I6.1 : monotone croissante — ajouter une hypothèse ≥ bits actifs.
        Invariant I6.5 : compute_union([h]) == h

        Args:
            hypotheses : liste de tenseurs (n,) bool

        Returns:
            (n,) bool
        """
        if not hypotheses:
            raise ValueError("compute_union() requiert au moins une hypothèse")
        if len(hypotheses) == 1:
            return hypotheses[0].clone()
        stacked = torch.stack(hypotheses, dim=0)  # (|H|, n)
        return stacked.any(dim=0)                  # (n,)

    def consensus(self, unions: List[torch.BoolTensor]) -> torch.BoolTensor:
        """
        Déf. 6.2 : C = AND booléen des unions de toutes les colonnes.

        Thm 6.1 : supp(C) = ∩ supp(U^(c))
        Invariant I6.2 : monotone décroissante — ajouter une colonne ≤ bits actifs.
        Invariant I6.3 : résultat identique à parallel_consensus.

        Args:
            unions : liste de K tenseurs (n,) bool, un par colonne

        Returns:
            (n,) bool
        """
        if not unions:
            raise ValueError("consensus() requiert au moins une union")
        if len(unions) == 1:
            return unions[0].clone()
        stacked = torch.stack(unions, dim=0)  # (K, n)
        return stacked.all(dim=0)              # (n,)

    def parallel_consensus(self, unions: List[torch.BoolTensor]) -> torch.BoolTensor:
        """
        Cor. 6.4 : réduction en arbre binaire → O(log K) étapes de profondeur.
        En pratique sur GPU : torch.stack().all(0) est O(n log K) en parallèle.

        Invariant I6.3 : résultat identique à consensus().
        """
        if not unions:
            raise ValueError("parallel_consensus() requiert au moins une union")
        # torch.stack + .all() exploite le parallélisme CUDA nativement
        stacked = torch.stack(unions, dim=0)
        return stacked.all(dim=0)
