"""
MODULE 1 — SDRSpace
Définitions 1.1–1.5
"""

import torch
import torch.nn as nn


class SDRSpace(nn.Module):
    """
    Espace des représentations distribuées clairsemées (SDR).

    Un SDR valide est un vecteur booléen de dimension n avec exactement w bits actifs.
    Ce module est purement fonctionnel : aucun paramètre appris.

    Contrainte de représentation (Déf. 1.1) : ||x||₁ = w exactement.
    """

    def __init__(self, n: int, w: int):
        """
        Args:
            n : dimension du vecteur SDR (ex. 2048)
            w : nombre exact de bits actifs (ex. 40)
        """
        super().__init__()
        assert 0 < w < n, f"w={w} doit être dans (0, n={n})"
        self.n = n
        self.w = w
        self.s = w / n  # ratio de parcimonie (Déf. 1.2)

    # ── Opérateurs de base ────────────────────────────────────────────────────

    def overlap(self, x: torch.BoolTensor, y: torch.BoolTensor) -> torch.Tensor:
        """
        Déf. 1.3 : O(x, y) = |supp(x) ∩ supp(y)|

        Invariant I1.1 : symétrique — overlap(x,y) == overlap(y,x)
        Invariant I1.2 : overlap(x,x) == w pour tout SDR valide
        Invariant I1.3 : résultat ∈ [0, min(||x||₁, ||y||₁)]
        """
        return (x & y).sum(dim=-1)

    def is_match(self, x: torch.BoolTensor, y: torch.BoolTensor,
                 theta: int) -> torch.BoolTensor:
        """
        Déf. 1.4 : correspondance positive si O(x, y) ≥ θ
        """
        return self.overlap(x, y) >= theta

    def union(self, vs: torch.BoolTensor) -> torch.BoolTensor:
        """
        Déf. 1.5 : U_i = max_m v^(m)_i  (OR booléen sur M hypothèses)

        Invariant I1.4 : union(vs).sum() ∈ [w, M*w]
        Note : la parcimonie n'est PAS garantie après une union.
                Appliquer top_w() si nécessaire.

        Args:
            vs : (M, n) bool — M hypothèses SDR

        Returns:
            (n,) bool
        """
        if vs.ndim == 1:
            return vs.clone()
        if vs.shape[0] == 0:
            raise ValueError("union() requiert au moins un SDR (liste vide reçue)")
        return vs.any(dim=0)

    def top_w(self, scores: torch.Tensor) -> torch.BoolTensor:
        """
        Réenforce la contrainte de parcimonie ||x||₁ = w après une opération
        qui aurait pu la violer (union, perturbation, etc.).

        Utilise top-k sur des scores flottants → sélectionne exactement w indices.
        """
        _, indices = torch.topk(scores.float(), self.w)
        out = torch.zeros(self.n, dtype=torch.bool, device=scores.device)
        out[indices] = True
        return out

    def random_sdr(self, device=None) -> torch.BoolTensor:
        """
        Génère un SDR aléatoire valide avec exactement w bits actifs.
        Utile pour les tests.
        """
        indices = torch.randperm(self.n, device=device)[:self.w]
        sdr = torch.zeros(self.n, dtype=torch.bool, device=device)
        sdr[indices] = True
        return sdr

    def false_positive_prob(self, theta: int, n: int, w: int) -> float:
        """
        Prop. 1.2 : probabilité qu'un SDR aléatoire dépasse θ par hasard.
        P(O(x,y) ≥ θ) ≈ Σ_{b=θ}^{w} C(w,b)·C(n-w, w-b) / C(n,w)

        Invariant I1.5 : décroissante en θ.
        """
        from math import comb
        total = comb(n, w)
        if total == 0:
            return 0.0
        prob = sum(
            comb(w, b) * comb(n - w, w - b)
            for b in range(theta, w + 1)
            if w - b <= n - w  # condition combinatoire
        ) / total
        return min(prob, 1.0)

    def forward(self, x: torch.BoolTensor) -> torch.BoolTensor:
        """Passe-through — SDRSpace est fonctionnel, pas de transformation forward."""
        return x
