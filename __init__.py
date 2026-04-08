"""
Colonne Corticale — Package PyTorch, Théorie des Mille Cerveaux
===============================================================
Baptiste Caillerie, 7 avril 2026

Modules (ordre de dépendance) :
  sdr_space        — Déf. 1.1–1.5
  spatial_pooler   — Déf. 2.1–2.6, Thm 2.1–2.3
  layer6b          — Déf. 4.1–4.3, Thm 4.1–4.2
  grid_cells       — Déf. 3.1–3.4, Thm 3.1–3.2
  displacement     — Déf. 5.1–5.3, Thm 5.1–5.2
  consensus        — Déf. 6.1–6.2, Thm 6.1–6.3
  cortical_column  — orchestrateur complet
"""

from .sdr_space import SDRSpace
from .spatial_pooler import SpatialPooler
from .layer6b import Layer6bTransformer
from .grid_cells import GridCellModule, GridCellNetwork
from .displacement import DisplacementAlgebra
from .consensus import MultiColumnConsensus
from .cortical_column import CorticalColumn

__all__ = [
    "SDRSpace",
    "SpatialPooler",
    "Layer6bTransformer",
    "GridCellModule",
    "GridCellNetwork",
    "DisplacementAlgebra",
    "MultiColumnConsensus",
    "CorticalColumn",
]
