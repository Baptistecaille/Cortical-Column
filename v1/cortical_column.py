"""
ASSEMBLAGE — CorticalColumn
Cycle complet : cortical_column_step
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from .sdr_space import SDRSpace
    from .spatial_pooler import SpatialPooler
    from .layer6b import Layer6bTransformer
    from .grid_cells import GridCellNetwork
    from .displacement import DisplacementAlgebra
    from .consensus import MultiColumnConsensus
except ImportError:
    from sdr_space import SDRSpace
    from spatial_pooler import SpatialPooler
    from layer6b import Layer6bTransformer
    from grid_cells import GridCellNetwork
    from displacement import DisplacementAlgebra
    from consensus import MultiColumnConsensus


class CorticalColumn(nn.Module):
    """
    Colonne corticale complète — assemble les 6 modules en un cycle de traitement.

    Cycle d'un pas (step) :
      1. SpatialPooler      : I_raw → C (SDR encodé)
      2. Layer6bTransformer : v_ego → v_allo (transformation ego→allo)
      3. GridCellNetwork    : v_allo → L_t (état de localisation)
      4. DisplacementAlgebra: (L_t, L_ref) → D (déplacement optionnel)
      5. SDRSpace           : matching C contre la mémoire locale
      6. MultiColumnConsensus : agrégation inter-colonnes (requiert les autres colonnes)
    """

    def __init__(
        self,
        N_in: int = 1024,
        N_mc: int = 2048,
        sdr_w: int = 40,
        s: float = 0.02,
        d: int = 2,
        K_grid: int = 4,
        K_columns: int = 4,
        N_orientations: int = 360,
    ):
        super().__init__()
        self.sdr_space = SDRSpace(n=N_mc, w=sdr_w)
        self.spatial_pooler = SpatialPooler(N_in=N_in, N_mc=N_mc, s=s)
        self.l6b = Layer6bTransformer(d=d, N_orientations=N_orientations)
        self.grid_net = GridCellNetwork(K=K_grid, d=d)
        self.displacement = DisplacementAlgebra(dims=[d] * K_grid)
        self.consensus = MultiColumnConsensus(n=N_mc, K=K_columns)

    def step(
        self,
        I_raw: torch.BoolTensor,
        v_ego: torch.Tensor,
        delta_orientation: int,
        local_hypotheses: Optional[List[torch.BoolTensor]] = None,
        learn: bool = True,
    ):
        """
        Cycle complet d'un pas de temps.

        Args:
            I_raw              : (N_in,) bool  — entrée sensorielle brute
            v_ego              : (d,)   float  — vitesse égocentrique
            delta_orientation  : int            — changement d'orientation discret
            local_hypotheses   : liste de SDR précédents pour le vote
            learn              : si True, effectue la mise à jour hebbienne

        Returns:
            C       : (N_mc,) bool  — SDR encodé
            L_t     : (K*d,) float  — état de localisation courant
            union_h : (N_mc,) bool  — union des hypothèses locales
        """
        # ── Étape 1 : Encodage spatial ──────────────────────────────────────
        C = self.spatial_pooler(I_raw)
        if learn:
            self.spatial_pooler.hebbian_update(I_raw, C)
            self.spatial_pooler.update_duty_cycle(C)

        # ── Étape 2 : Transformation thalamique L6b ─────────────────────────
        self.l6b.update_orientation(delta_orientation)
        v_allo = self.l6b.transform(v_ego)

        # ── Étape 3 : Intégration de chemin ─────────────────────────────────
        self.grid_net.integrate_all(v_allo, dt=1.0)
        L_t = self.grid_net.get_location_state()

        # ── Étape 4 : Hypothèses locales + union ─────────────────────────────
        hypotheses = local_hypotheses or []
        hypotheses = hypotheses + [C]
        union_h = self.consensus.compute_union(hypotheses)

        return C, L_t, union_h
