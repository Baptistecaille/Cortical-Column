"""
test_path_independence.py
=========================
Test du Thm 3.1 : intégration de chemin conservative (path-independence).

Thm 3.1 : φ(T) = φ_0 + A · ∫v dt
          Si ∫v dt = 0 (circuit fermé), alors φ(T) = φ_0.

La propriété testée ici est que pour tout circuit fermé, l'erreur de fermeture
φ_final - φ_0 est nulle (aux erreurs numériques float32 près).

Conventions :
    - Les phases vivent sur un tore [0,1)^D — distance torique composante par composante.
    - D = 64, M = 8 modules (valeurs CLAUDE.md), donc d = 8 par module.
    - Intégration via GridCellNetwork.integrate_all(v, dt=1.0).
"""

import sys
import os
import math
import torch
from typing import List, Tuple

# Ajoute le répertoire parent au chemin d'import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortical_column import GridCellNetwork


# ─── Utilitaires ─────────────────────────────────────────────────────────────

def torus_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Distance L2 entre deux points sur le tore [0,1)^D.

    Pour chaque composante i : dist_i = min(|a_i - b_i|, 1 - |a_i - b_i|)
    Puis norme L2 sur l'ensemble des composantes.

    Déf. 3.2 : les phases sont sur un tore — il faut tenir compte du wrapping.

    Args:
        a : (D,) float — phase A ∈ [0, 1)^D
        b : (D,) float — phase B ∈ [0, 1)^D

    Returns:
        Scalaire — distance torique L2 entre a et b.
    """
    diff = torch.abs(a - b)
    # Pour chaque composante, prendre le chemin le plus court sur le cercle
    wrapped_diff = torch.min(diff, 1.0 - diff)
    return torch.norm(wrapped_diff)


def run_circuit(
    grid_net: GridCellNetwork,
    velocities: List[torch.Tensor],
    dt: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exécute un circuit de déplacements et retourne l'état initial, final et l'erreur.

    Thm 3.1 : si sum(velocities) == 0, alors L_final ≈ L_0.

    Args:
        grid_net   : instance GridCellNetwork fraîchement réinitialisée
        velocities : liste de vecteurs de vitesse (d,) float
        dt         : pas de temps (Déf. 3.2)

    Returns:
        L_0     : (K*d,) float — état initial
        L_final : (K*d,) float — état après tous les déplacements
        err     : scalaire float — erreur de fermeture torique L2
    """
    # Enregistrer l'état initial
    L_0 = grid_net.get_location_state().clone()

    # Intégrer chaque déplacement du circuit
    for v in velocities:
        grid_net.integrate_all(v, dt=dt)

    # Récupérer l'état final
    L_final = grid_net.get_location_state().clone()

    # Erreur de fermeture en tenant compte du wrapping torique
    err = torus_dist(L_final, L_0)

    return L_0, L_final, err


# ─── Définition des circuits ──────────────────────────────────────────────────

def make_square_circuit(d: int = 2) -> List[torch.Tensor]:
    """
    Circuit carré : 4 pas de longueur 1.0 dans les directions N, E, S, O.

    Vecteurs : [0,1], [1,0], [0,-1], [-1,0]  (dans les 2 premières composantes)
    Les dimensions supplémentaires (d > 2) sont nulles.

    Déf. 3.3 : v_allo ∈ ℝ^d — les composantes supplémentaires restent à zéro.

    Returns:
        Liste de 4 tenseurs (d,) float dont la somme est exactement zéro.
    """
    assert d >= 2, "Le circuit carré requiert d >= 2"

    def vec(x: float, y: float) -> torch.Tensor:
        """Crée un vecteur de dimension d avec les deux premières composantes données."""
        v = torch.zeros(d, dtype=torch.float32)
        v[0] = x
        v[1] = y
        return v

    # N, E, S, O — somme = zéro par construction
    return [
        vec(0.0,  1.0),   # Nord
        vec(1.0,  0.0),   # Est
        vec(0.0, -1.0),   # Sud
        vec(-1.0, 0.0),   # Ouest
    ]


def make_triangle_circuit(d: int = 2) -> List[torch.Tensor]:
    """
    Circuit triangulaire équilatéral : 3 pas dont la somme est zéro.

    Décomposition :
        v_0 = [1, 0]
        v_1 = [-cos(60°), sin(60°)]  = [-0.5,  √3/2]
        v_2 = [-cos(60°), -sin(60°)] = [-0.5, -√3/2]
    Vérification : v_0 + v_1 + v_2 = [0, 0] exactement.

    Returns:
        Liste de 3 tenseurs (d,) float dont la somme est exactement zéro.
    """
    assert d >= 2, "Le circuit triangulaire requiert d >= 2"

    cos60 = 0.5
    sin60 = math.sqrt(3) / 2

    def vec(x: float, y: float) -> torch.Tensor:
        v = torch.zeros(d, dtype=torch.float32)
        v[0] = x
        v[1] = y
        return v

    return [
        vec(1.0,    0.0),
        vec(-cos60,  sin60),
        vec(-cos60, -sin60),
    ]


def make_random_circuit(N: int = 20, d: int = 2, seed: int = 42) -> List[torch.Tensor]:
    """
    Circuit aléatoire : N-1 pas aléatoires + 1 pas de fermeture exact.

    Le dernier vecteur est calculé pour que la somme totale soit exactement zéro :
        v_closing = -sum(v_1, ..., v_{N-1})

    Args:
        N    : nombre total de pas (dont le dernier ferme le circuit)
        d    : dimension des vecteurs
        seed : graine pour la reproductibilité

    Returns:
        Liste de N tenseurs (d,) float dont la somme est exactement zéro.
    """
    assert N >= 2, "Un circuit requiert au moins 2 pas"

    torch.manual_seed(seed)
    velocities = []

    # N-1 pas aléatoires dans [-1, 1]^d
    for _ in range(N - 1):
        v = torch.rand(d, dtype=torch.float32) * 2 - 1  # uniforme dans [-1, 1]^d
        velocities.append(v)

    # Pas de fermeture : annule exactement la dérive accumulée
    v_closing = -torch.stack(velocities).sum(dim=0)
    velocities.append(v_closing)

    # Vérification interne : la somme doit être nulle à la précision float32
    total = torch.stack(velocities).sum(dim=0)
    assert torch.allclose(total, torch.zeros(d), atol=1e-5), \
        f"Erreur de construction du circuit aléatoire : somme = {total}"

    return velocities


# ─── Programme principal ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dimensions cibles issues de CLAUDE.md :
    #   D = 64 (dimension totale de la phase grid cell)
    #   M = 8  (nombre de modules grid cells)
    #   => d = D / M = 8 composantes par module
    M: int = 8   # nombre de modules
    d: int = 8   # dimension par module (D/M = 64/8)
    TOLERANCE: float = 1e-4  # tolérance pour les assertions (erreurs float32)

    print("=" * 60)
    print("Test Thm 3.1 — Intégration de chemin conservative")
    print(f"Configuration : M={M} modules, d={d} dim/module "
          f"(D_total={M*d})")
    print("=" * 60)

    # ── Circuit 1 : Carré ─────────────────────────────────────────────────────
    print("\n[1/3] Circuit carré (4 pas, directions N-E-S-O)")

    grid_net = GridCellNetwork(K=M, d=d)
    grid_net.reset_all()  # φ_0 = 0 pour tous les modules

    square_vels = make_square_circuit(d=d)
    print(f"      Vecteurs : {[v[:2].tolist() for v in square_vels]}  (2 premières composantes)")

    L0_sq, Lf_sq, err_sq = run_circuit(grid_net, square_vels)

    print(f"      L_0     (extrait) : {L0_sq[:4].tolist()}")
    print(f"      L_final (extrait) : {Lf_sq[:4].tolist()}")
    print(f"      Erreur de fermeture (distance torique L2) : {err_sq.item():.2e}")

    if err_sq >= TOLERANCE:
        print(f"  !! ECHEC : err={err_sq.item():.6f} >= tolerance={TOLERANCE}")
        print(f"     L_0     = {L0_sq.tolist()}")
        print(f"     L_final = {Lf_sq.tolist()}")
        print(f"     diff    = {(Lf_sq - L0_sq).tolist()}")
    else:
        print(f"      OK — err < {TOLERANCE}")

    assert err_sq < TOLERANCE, \
        f"Circuit carré : erreur de fermeture {err_sq.item():.6f} >= {TOLERANCE}"

    # ── Circuit 2 : Triangle équilatéral ─────────────────────────────────────
    print("\n[2/3] Circuit triangulaire équilatéral (3 pas)")

    grid_net.reset_all()  # réinitialiser l'état

    triangle_vels = make_triangle_circuit(d=d)
    print(f"      Vecteurs : {[v[:2].tolist() for v in triangle_vels]}  (2 premières composantes)")

    L0_tr, Lf_tr, err_tr = run_circuit(grid_net, triangle_vels)

    print(f"      L_0     (extrait) : {L0_tr[:4].tolist()}")
    print(f"      L_final (extrait) : {Lf_tr[:4].tolist()}")
    print(f"      Erreur de fermeture (distance torique L2) : {err_tr.item():.2e}")

    if err_tr >= TOLERANCE:
        print(f"  !! ECHEC : err={err_tr.item():.6f} >= tolerance={TOLERANCE}")
        print(f"     L_0     = {L0_tr.tolist()}")
        print(f"     L_final = {Lf_tr.tolist()}")
        print(f"     diff    = {(Lf_tr - L0_tr).tolist()}")
    else:
        print(f"      OK — err < {TOLERANCE}")

    assert err_tr < TOLERANCE, \
        f"Circuit triangulaire : erreur de fermeture {err_tr.item():.6f} >= {TOLERANCE}"

    # ── Circuit 3 : Aléatoire (N=20 pas) ────────────────────────────────────
    print("\n[3/3] Circuit aléatoire (N=20 pas, seed=42)")

    grid_net.reset_all()

    random_vels = make_random_circuit(N=20, d=d, seed=42)
    print(f"      Nombre de pas : {len(random_vels)}")
    print(f"      Somme totale  : {torch.stack(random_vels).sum(dim=0)[:2].tolist()} ... (≈ zéro)")

    L0_rnd, Lf_rnd, err_rnd = run_circuit(grid_net, random_vels)

    print(f"      L_0     (extrait) : {L0_rnd[:4].tolist()}")
    print(f"      L_final (extrait) : {Lf_rnd[:4].tolist()}")
    print(f"      Erreur de fermeture (distance torique L2) : {err_rnd.item():.2e}")

    if err_rnd >= TOLERANCE:
        print(f"  !! ECHEC : err={err_rnd.item():.6f} >= tolerance={TOLERANCE}")
        print(f"     L_0     = {L0_rnd.tolist()}")
        print(f"     L_final = {Lf_rnd.tolist()}")
        print(f"     diff    = {(Lf_rnd - L0_rnd).tolist()}")
    else:
        print(f"      OK — err < {TOLERANCE}")

    assert err_rnd < TOLERANCE, \
        f"Circuit aléatoire : erreur de fermeture {err_rnd.item():.6f} >= {TOLERANCE}"

    # ── Récapitulatif ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Récapitulatif des erreurs de fermeture (Thm 3.1)")
    print(f"  Circuit carré      : {err_sq.item():.2e}")
    print(f"  Circuit triangulaire : {err_tr.item():.2e}")
    print(f"  Circuit aléatoire  : {err_rnd.item():.2e}")
    print(f"  Tolérance          : {TOLERANCE:.2e}")
    print("=" * 60)
    print("Tous les tests sont passés. Thm 3.1 vérifié.")
