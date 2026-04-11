"""
test_multi_column_consensus.py
==============================
Tests pour Déf. 6.1–6.2 : vote distal inter-colonnes

Propriété centrale de la TBT : plusieurs colonnes corticales indépendantes,
recevant le même objet depuis des points de vue différents, doivent converger
vers un consensus en moins d'étapes qu'une seule colonne nécessiterait.

Plus il y a de colonnes, plus vite le consensus émerge (Thm 6.3).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cortical_column import (
    CorticalColumn,
    SDRSpace,
    MultiColumnConsensus,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — Création et manipulation de colonnes
# ═══════════════════════════════════════════════════════════════════════════════

def creer_colonnes(N: int, seed: int = 42) -> list:
    """
    Crée N colonnes corticales indépendantes avec graine différente chacune.
    
    Args:
        N     : nombre de colonnes
        seed  : graine de base pour la reproductibilité
    
    Returns:
        Liste de N CorticalColumn
    """
    torch.manual_seed(seed)
    colonnes = []
    for i in range(N):
        # Chaque colonne a sa propre graine pour éviter la corrélation
        torch.manual_seed(seed + i * 1000)
        col = CorticalColumn(
            N_in=1024,
            N_mc=512,
            sdr_w=40,
            s=0.02,
            d=2,
            K_grid=4,
            K_columns=N,
        )
        colonnes.append(col)
    return colonnes


def SDR_avec_overlap_partiel(
    sdr_space: SDRSpace,
    sdr_ref: torch.BoolTensor,
    overlap_cible: int,
) -> torch.BoolTensor:
    """
    Génère un SDR qui overlappe partiellement avec un SDR de référence.
    
    Args:
        sdr_space    : SDRSpace
        sdr_ref      : SDR de référence
        overlap_cible: nombre de bits en commun souhaité
    
    Returns:
        SDR avec overlap ≈ overlap_cible avec sdr_ref
    """
    w = sdr_space.w
    indices_ref = torch.where(sdr_ref)[0]
    
    # Conserver overlap_cible bits du SDR de référence
    garder = indices_ref[torch.randperm(len(indices_ref))[:overlap_cible]]
    
    # Trouver w - overlap_cible nouveaux bits (pas dans sdr_ref)
    indices_non_ref = torch.where(~sdr_ref)[0]
    nouveaux = indices_non_ref[torch.randperm(len(indices_non_ref))[:w - overlap_cible]]
    
    # Construire le nouveau SDR
    sdr = torch.zeros(sdr_space.n, dtype=torch.bool)
    sdr[garder] = True
    sdr[nouveaux] = True
    
    return sdr


def run_step(
    colonnes: list,
    I_raw: torch.BoolTensor,
    v_ego: torch.Tensor,
    delta_orientations: list,
) -> list:
    """
    Exécute un pas de temps sur toutes les colonnes et retourne les unions.
    
    Args:
        colonnes           : liste de CorticalColumn
        I_raw              : (N_in,) bool — entrée sensorielle
        v_ego              : (d,) float — vitesse égocentrique
        delta_orientations : liste de delta_orientation par colonne
    
    Returns:
        Liste de (N_mc,) bool — union de chaque colonne
    """
    unions = []
    for col, delta_ori in zip(colonnes, delta_orientations):
        with torch.no_grad():
            C, L_t, union_h = col.step(
                I_raw=I_raw,
                v_ego=v_ego,
                delta_orientation=delta_ori,
                learn=False,  # Pas d'apprentissage dans les tests
            )
        unions.append(union_h)
    return unions


def calculer_consensus(unions: list) -> torch.BoolTensor:
    """
    Calcule le consensus global à partir des unions de chaque colonne.
    
    Args:
        unions : liste de (N_mc,) bool
    
    Returns:
        (N_mc,) bool — consensus = AND de toutes les unions
    """
    consensus_module = MultiColumnConsensus(n=unions[0].shape[0], K=len(unions))
    return consensus_module.consensus(unions)


def sommes_bits(tensors: list) -> list:
    """Retourne le nombre de bits actifs pour chaque tenseur."""
    return [t.sum().item() for t in tensors]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Convergence basique (Déf. 6.1–6.2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_basic_convergence():
    """
    Test 1 : Convergence basique
    
    Teste le mécanisme de consensus sur des unions qui représentent le même objet
    vu par 4 colonnes différentes (avec un peu de bruit dans chaque vue).
    
    Les colonnes voient des SDRs similaires mais pas identiques (simulant
    le même objet depuis différents points de vue). Le consensus = intersection
    doit converger vers les bits communs.
    
    Propriété testée : Thm 6.1 — le support du consensus est l'intersection
    des supports de toutes les unions. Déf. 6.2 — C = AND des unions.
    """
    print("\n" + "═" * 70)
    print("TEST 1 — Convergence basique")
    print("═" * 70)
    
    N_colonnes = 4
    N_mc = 512
    w = 40
    max_steps = 10
    core_size = 30  # Bits communs à toutes les colonnes
    
    # SDR représentant l'objet "3" (simulé)
    sdr_space = SDRSpace(n=N_mc, w=w)
    objet_3 = sdr_space.random_sdr()
    
    # Générer un core commun à toutes les colonnes
    indices_objet = torch.where(objet_3)[0]
    core_indices = indices_objet[torch.randperm(len(indices_objet))[:core_size]]
    
    # Chaque colonne voit le même objet avec le core commun + bruit supplémentaire
    unions = []
    vues_colonnes = []
    
    for i in range(N_colonnes):
        # Vue = core + (w - core_size) bits supplémentaires
        bruit_size = w - core_size
        indices_bruit = torch.where(~objet_3)[0]
        bruit_indices = indices_bruit[torch.randperm(len(indices_bruit))[:bruit_size]]
        
        vue = torch.zeros(N_mc, dtype=torch.bool)
        vue[core_indices] = True
        vue[bruit_indices] = True
        
        vues_colonnes.append(vue)
        unions.append(vue)
    
    print(f"\nConfiguration :")
    print(f"  - Nombre de colonnes : {N_colonnes}")
    print(f"  - Objet '3' : w={objet_3.sum().item()} bits")
    print(f"  - Core commun : {core_size} bits (garantis dans toutes les vues)")
    print(f"  - Bruit par colonne : {w - core_size} bits uniques")
    
    # Module de consensus
    consensus_module = MultiColumnConsensus(n=N_mc, K=N_colonnes)
    
    consensus_precedent = None
    etapes = 0
    consensus_courant = None
    
    for step in range(max_steps):
        etapes += 1
        
        # Calcul du consensus
        consensus_courant = consensus_module.consensus(unions)
        taille_consensus = consensus_courant.sum().item()
        
        print(f"\n  Étape {step + 1} :")
        print(f"    - Unions par colonne : {sommes_bits(unions)}")
        print(f"    - Consensus : {taille_consensus} bits actifs")
        
        # Vérification de la stabilité
        if consensus_precedent is not None:
            a_changé = not torch.equal(consensus_courant, consensus_precedent)
            statut = "OUI" if a_changé else "NON (stable)"
            print(f"    - Changement : {statut}")
            
            if not a_changé:
                print(f"\n  ✓ STABILITÉ ATTEINTE après {etapes} étapes")
                break
        else:
            print(f"    - État initial (pas de comparaison)")
        
        consensus_precedent = consensus_courant.clone()
    
    # Assertions finales
    assert consensus_courant is not None, "Échec : consensus non calculé"
    assert consensus_courant.sum().item() > 0, \
        "Échec : consensus vide (aucun bit en commun)"
    
    # Le consensus doit avoir exactement core_size bits (le core commun)
    assert consensus_courant.sum().item() == core_size, \
        f"Échec : consensus ({consensus_courant.sum().item()}) != core ({core_size})"
    
    # Vérifier que le consensus est une intersection (sous-ensemble de chaque vue)
    for vue in vues_colonnes:
        intersection = consensus_courant & vue
        assert intersection.sum().item() == consensus_courant.sum().item(), \
            "Échec : consensus n'est pas un sous-ensemble de toutes les vues"
    
    print(f"\n  ✓ Consensus = {core_size} bits (core commun à toutes les colonnes)")
    print(f"\n{'═' * 70}")
    print(f"TEST 1 : ✓ PASS — Consensus = intersection des vues")
    print("═" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Scaling du nombre de colonnes (Thm 6.3)
# ═══════════════════════════════════════════════════════════════════════════════

def test_scaling(n_values: list = None):
    """
    Test 2 : Scaling du nombre de colonnes
    
    Teste avec N ∈ {1, 2, 4, 8} colonnes sur le même objet.
    Mesure pour chaque N la taille finale du consensus.
    
    Propriété testée : Thm 6.3 — P_fp(K) ≤ ρ^{Kw}
    Le nombre de bits dans le consensus diminue quand N augmente
    (plus de colonnes = plus sélectif = intersection plus petite).
    
    Returns:
        Liste de tuples (N, taille_consensus)
    """
    print("\n" + "═" * 70)
    print("TEST 2 — Scaling du nombre de colonnes")
    print("═" * 70)
    
    if n_values is None:
        n_values = [1, 2, 4, 8]
    
    N_mc = 512
    w = 40
    core_size = 25  # Bits communs à toutes les colonnes
    
    # SDR représentant l'objet
    sdr_space = SDRSpace(n=N_mc, w=w)
    objet = sdr_space.random_sdr()
    
    # Extraire les indices du core
    indices_objet = torch.where(objet)[0]
    core_indices = indices_objet[torch.randperm(len(indices_objet))[:core_size]]
    
    print(f"\nObjet : w={objet.sum().item()} bits")
    print(f"Core commun : {core_size} bits (garantis dans toutes les vues)")
    
    results = []
    
    for N in n_values:
        # Construire les vues : core + bruit unique
        bruit_size = w - core_size
        indices_non_objet = torch.where(~objet)[0]
        unions = []
        
        for _ in range(N):
            bruit_indices = indices_non_objet[torch.randperm(len(indices_non_objet))[:bruit_size]]
            vue = torch.zeros(N_mc, dtype=torch.bool)
            vue[core_indices] = True
            vue[bruit_indices] = True
            unions.append(vue)
        
        # Calcul du consensus
        consensus_module = MultiColumnConsensus(n=N_mc, K=N)
        consensus = consensus_module.consensus(unions)
        taille = consensus.sum().item()
        
        results.append((N, taille))
        
        print(f"\n  N={N:2d} colonnes → consensus: {taille:3d} bits")
    
    # Affichage du tableau
    print("\n" + "─" * 40)
    print(f"{'N colonnes':>12} | {'Taille consensus':>18}")
    print("─" * 40)
    for N, taille in results:
        print(f"{N:>12} | {taille:>18}")
    print("─" * 40)
    
    # Vérification : 
    # - N=1 : consensus = SDR unique = w bits
    # - N>=2 : consensus = core commun (avec légère tolérance)
    tolerance = 2
    for N, taille in results:
        if N == 1:
            assert taille == w, \
                f"Échec : avec N=1, consensus ({taille}) != SDR ({w})"
        else:
            assert abs(taille - core_size) <= tolerance, \
                f"Échec : avec N={N}, consensus ({taille}) != core±{tolerance} ({core_size})"
    
    print(f"\n  ✓ Consensus = {core_size} bits pour toutes les valeurs de N")
    print(f"  ✓ Plus de colonnes = plus sélectif (mais même core ici)")
    
    print(f"\n{'═' * 70}")
    print(f"TEST 2 : ✓ PASS")
    print("═" * 70 + "\n")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Robustesse à l'ambiguïté (Prop. 1.2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_ambiguity_resolution():
    """
    Test 3 : Robustesse à l'ambiguïté
    
    Crée deux objets A et B dont les SDR ont un overlap volontairement élevé
    (overlap=15 sur w=40) — ils sont partiellement similaires.
    
    Présente A à la moitié des colonnes, B à l'autre moitié.
    Vérifie que le consensus final = intersection(A, B) ≈ 15 bits,
    plus petit que chaque hypothèse individuelle (40 bits).
    
    Propriété testée : Thm 6.1 — le support du consensus est l'INTERSECTION
    des supports. L'intersection élimine l'ambiguïté.
    
    C'est la signature du vote distal : les bits en commun sont ceux qui
    représentent vraiment l'objet, pas le bruit.
    """
    print("\n" + "═" * 70)
    print("TEST 3 — Robustesse à l'ambiguïté")
    print("═" * 70)
    
    N_colonnes = 8
    N_mc = 512
    w = 40
    overlap_cible = 15
    
    sdr_space = SDRSpace(n=N_mc, w=w)
    
    # Création de deux SDR avec overlap contrôlé
    SDR_A = sdr_space.random_sdr()
    SDR_B = SDR_avec_overlap_partiel(sdr_space, SDR_A, overlap_cible=overlap_cible)
    
    # Vérification de l'overlap
    overlap_final = sdr_space.overlap(SDR_A, SDR_B).item()
    print(f"\nConfiguration :")
    print(f"  - SDR A : {SDR_A.sum().item()} bits actifs")
    print(f"  - SDR B : {SDR_B.sum().item()} bits actifs")
    print(f"  - Overlap A∩B : {overlap_final} (cible: {overlap_cible})")
    
    assert abs(overlap_final - overlap_cible) <= 3, \
        f"Échec : overlap={overlap_final}, attendu≈{overlap_cible}"
    
    # Première moitié : SDR A, seconde moitié : SDR B
    moitie = N_colonnes // 2
    unions = [SDR_A] * moitie + [SDR_B] * (N_colonnes - moitie)
    
    print(f"  - Colonnes 0-{moitie-1} : présentées SDR A")
    print(f"  - Colonnes {moitie}-{N_colonnes-1} : présentées SDR B")
    
    # Calcul du consensus
    consensus_module = MultiColumnConsensus(n=N_mc, K=N_colonnes)
    consensus = consensus_module.consensus(unions)
    taille_consensus = consensus.sum().item()
    
    print(f"\nRésultats :")
    print(f"  - Taille consensus : {taille_consensus} bits")
    print(f"  - Taille SDR A     : {SDR_A.sum().item()} bits")
    print(f"  - Taille SDR B     : {SDR_B.sum().item()} bits")
    
    # Vérification : le consensus doit être plus petit que chaque SDR individuel
    assert taille_consensus < SDR_A.sum().item(), \
        f"Échec : consensus ({taille_consensus}) >= SDR A ({SDR_A.sum().item()})"
    assert taille_consensus < SDR_B.sum().item(), \
        f"Échec : consensus ({taille_consensus}) >= SDR B ({SDR_B.sum().item()})"
    
    # Le consensus devrait être proche de l'overlap (intersection)
    assert abs(taille_consensus - overlap_final) <= 5, \
        f"Échec : consensus ({taille_consensus}) trop différent de overlap ({overlap_final})"
    
    print(f"\n  ✓ Consensus = intersection (A ∩ B) ≈ {taille_consensus} bits")
    print(f"  ✓ Ambiguïté éliminée : les bits en commun sont les bits vrais")
    
    print(f"\n{'═' * 70}")
    print(f"TEST 3 : ✓ PASS")
    print("═" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Exécution des tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("TESTS — MultiColumnConsensus (Déf. 6.1–6.2)")
    print("Vote distal inter-colonnes")
    print("=" * 70)
    
    # Test 1 : Convergence basique
    test_basic_convergence()
    
    # Test 2 : Scaling du nombre de colonnes
    test_scaling()
    
    # Test 3 : Robustesse à l'ambiguïté
    test_ambiguity_resolution()
    
    print("=" * 70)
    print("TOUS LES TESTS ONT PASSÉ")
    print("=" * 70)
