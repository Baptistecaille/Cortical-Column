"""
CorticalColumn v3 — Minicolonne étendue
========================================
Extensions par rapport à v2 :
  1. Hétérogénéité transcriptomique : 11 populations (PV basket / chandelier,
     SST Martinotti / non-Martinotti, VIP1 / VIP2, NGF)
  2. Couplage thalamus-NRT-L6 complet avec canal T-type (h_T, τ=100 ms)
  3. Règle STDP tri-factorielle gated par P_Ca et VIP1

Vecteur d'état (14 variables) :
  [0]  r_L4    — stellaires épineuses L4
  [1]  r_Ls    — pyramidaux L2/3  (feedforward gamma)
  [2]  r_Ld    — pyramidaux L5   (feedback beta)
  [3]  r_L6    — corticothalamiques L6
  [4]  r_PVb   — PV basket       (gain divisif rapide, τ=5 ms)
  [5]  r_PVc   — PV chandelier   (veto spike sur AIS, τ=7 ms)
  [6]  r_SSTm  — SST Martinotti  (inhibe apical en L1, τ=30 ms)
  [7]  r_SSTnm — SST non-Martinotti (contraste latéral L4, τ=20 ms)
  [8]  r_VIP1  — VIP type 1      (→ SST désinhibition, τ=15 ms)
  [9]  r_VIP2  — VIP type 2      (→ PVb faible, τ=15 ms)
  [10] r_NGF   — neurogliaform   (GABA-B lent L1, τ=40 ms)
  [11] r_TC    — relais thalamique (TC, τ=20 ms)
  [12] h_T     — inactivation T-type [0,1], τ=100 ms  — LENTE
  [13] r_NRT   — noyau réticulaire thalamique (NRT, τ=15 ms)

Références :
  Douglas & Martin (1991) — microcircuit canonique
  Bastos & Friston (2012) — codage prédictif, asymétrie gamma/bêta
  Mikulasch et al. (2022) — SNN-PC, règle tri-factorielle
  Kätzel et al. (2010)    — cartographie PV/SST par optogénétique
  Thomson et al. (2002)   — probabilités de connexion L4→L3, L3→L5
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# INDEX ET CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

CORTICAL = ['L4', 'Ls', 'Ld', 'L6', 'PVb', 'PVc', 'SSTm', 'SSTnm', 'VIP1', 'VIP2', 'NGF']
THALAMIC = ['TC', 'hT', 'NRT']
ALL_POPS = CORTICAL + THALAMIC

IDX = {k: i for i, k in enumerate(ALL_POPS)}
N   = len(ALL_POPS)      # 14 variables d'état
NC  = len(CORTICAL)      # 11 populations corticales

# Constantes de temps (ms) — biologiques
TAU_MS = {
    'L4':    15.0,
    'Ls':    20.0,
    'Ld':    25.0,
    'L6':    40.0,
    'PVb':    5.0,   # rapide !
    'PVc':    7.0,
    'SSTm':  30.0,
    'SSTnm': 20.0,
    'VIP1':  15.0,
    'VIP2':  15.0,
    'NGF':   40.0,
    'TC':    20.0,
    'hT':   100.0,   # lent — canaux T-type
    'NRT':   15.0,
}
TAU = torch.tensor([TAU_MS[k] for k in ALL_POPS])  # (14,)

R_MAX      = 100.0  # Hz — taux de décharge maximal
THETA_CA   = 0.15   # seuil plateau calcique (normalisé, ≈10 mV proxy)
LAM_BURST  = 2.5    # amplification burst gamma quand P_Ca=1


# ══════════════════════════════════════════════════════════════════════════════
# MATRICE DE CONNECTIVITÉ CORTICALE (11×11)
# ══════════════════════════════════════════════════════════════════════════════

def build_J() -> torch.Tensor:
    """
    Matrice J_cortex (11×11), convention J[post, pre].
    Positif = excitateur, négatif = inhibiteur.

    Règles structurelles (Thomson 2002, Kätzel 2010) :
      J[Ls, L4] > J[Ld, Ls] > J[L6, Ls]   (feedforward dominant)
      J[vertical] ≈ 15 × J[latéral]
      PVb reçoit fort drive thalamique (p=0.45)
      VIP1 → SSTm fort (désinhibition principale)
      VIP2 → PVb faible (w_{pvvip} asymétrique)
    """
    J = torch.zeros(NC, NC)
    i = IDX  # raccourci

    # ── Connexions excitatrices ──────────────────────────────────────────────
    J[i['L4'],    i['L4']]    =  0.30  # récurrence L4
    J[i['Ls'],    i['L4']]    =  1.50  # L4→L2/3  (plus forte FF, p=0.22)
    J[i['Ld'],    i['Ls']]    =  0.90  # L2/3→L5
    J[i['L6'],    i['Ls']]    =  0.40  # L2/3→L6
    J[i['L6'],    i['Ld']]    =  0.30  # L5→L6
    J[i['L6'],    i['L6']]    =  0.50  # récurrence L6 (intégration de chemin)
    J[i['PVb'],   i['L4']]    =  0.90  # L4→PVb fort (fenêtre 2-5 ms)
    J[i['PVb'],   i['Ls']]    =  0.50
    J[i['PVb'],   i['Ld']]    =  0.30
    J[i['PVc'],   i['L4']]    =  0.70  # L4→PVc (veto AIS)
    J[i['PVc'],   i['Ls']]    =  0.30
    J[i['SSTm'],  i['Ld']]    =  0.45  # L5→SSTm (activé par L5, pas L4)
    J[i['SSTm'],  i['Ls']]    =  0.20
    J[i['SSTnm'], i['L4']]    =  0.50  # L4→SSTnm (contraste latéral)
    J[i['NGF'],   i['Ls']]    =  0.20
    J[i['NGF'],   i['L6']]    =  0.10

    # ── Connexions inhibitrices ──────────────────────────────────────────────
    J[i['Ls'],    i['PVb']]   = -0.80  # PVb→L2/3 périsomatique (gain divisif)
    J[i['Ls'],    i['PVc']]   = -0.50  # PVc→L2/3 AIS (veto)
    J[i['Ls'],    i['SSTm']]  = -0.40  # SSTm→L2/3 dendrite apical
    J[i['Ls'],    i['NGF']]   = -0.20  # NGF→L2/3 GABA-B lent
    J[i['Ld'],    i['PVb']]   = -0.50  # PVb→L5
    J[i['Ld'],    i['SSTm']]  = -0.30  # SSTm→L5
    J[i['L4'],    i['SSTnm']] = -0.40  # SSTnm→L4 (contraste inter-colonnes)
    J[i['L4'],    i['L6']]    = -0.30  # L6→L4 feedback inhibiteur (proxy NRT)
    J[i['PVb'],   i['VIP2']]  = -0.20  # VIP2→PVb (désinhibition faible)
    J[i['SSTm'],  i['VIP1']]  = -0.80  # VIP1→SSTm (désinhibition principale)
    J[i['SSTnm'], i['VIP1']]  = -0.30  # VIP1→SSTnm

    return J


# ══════════════════════════════════════════════════════════════════════════════
# MINICOLONNE V3
# ══════════════════════════════════════════════════════════════════════════════

class MiniColumnV3:
    """
    Système dynamique à 14 variables.

    Invariants :
      I1 : r_k ∈ [0, R_MAX] pour k ∈ [0,10] ∪ {11, 13}
      I2 : h_T ∈ [0, 1]
      I3 : τ_PVb < τ_PVc < τ_Ls < τ_Ld < τ_L6 < τ_hT  (hiérarchie temporelle)
      I4 : P_Ca ∈ {0.0, 1.0}  (step function)
      I5 : J_s4_learned ≥ 0  (poids excitateur)
    """

    # Paramètres thalamocorticaux biologiques
    W_TC6    = 0.30  # L6→TC (métabotropique lent)
    W_TC_NRT = 0.70  # TC→NRT (glutamatergique)
    W_NRT_TC = 0.90  # NRT→TC (GABAergique, GABA-A+B)
    W_NRT_6  = 0.80  # L6→NRT (fort)
    G_T      = 0.50  # conductance canal T-type

    # Paramètres dendritiques
    G_B4    = 1.50  # basal ← L4
    G_ATOP  = 1.20  # apical ← μ
    G_BPVB  = 0.80  # basal ← PVb (inhibition périsomatique)
    G_ASST  = 0.60  # apical ← SSTm (filtre top-down)
    G_C     = 0.10  # couplage électrotonique basal-apical

    # Plasticité
    ETA     = 5e-4   # taux d'apprentissage STDP
    ETA_DEC = 5e-5   # décroissance homéostatique
    TAU_STDP = 20.0  # ms — fenêtre STDP

    def __init__(self, dt: float = 0.1):
        """
        Args:
            dt : pas d'intégration en ms (recommandé ≤ 0.1 ms pour stabilité PVb)
        """
        assert dt <= TAU_MS['PVb'] / 5, f"dt={dt}ms trop grand pour τ_PVb={TAU_MS['PVb']}ms"
        self.dt = dt

        # Matrice de connectivité corticale (apprise via J_s4)
        self.J = build_J()               # (11, 11) — modifiée à chaque pas

        # Poids L4→L2/3 appris par STDP (séparé de J pour suivi)
        self.J_s4 = torch.tensor(1.50)

        # Traces STDP (exponentielles)
        self.tr_pre  = torch.tensor(0.0)
        self.tr_post = torch.tensor(0.0)

        # Lésion L6 (réduit r_L6 à 0 si actif)
        self.lesion_L6 = False

    # ── Fonctions de transfert ───────────────────────────────────────────────

    @staticmethod
    def phi(x: torch.Tensor, beta: float = 0.10, theta: float = 0.0) -> torch.Tensor:
        """Sigmoïde bornée dans [0, R_MAX]. Éq. 1."""
        return R_MAX / (1.0 + torch.exp(-beta * (x - theta)))

    @staticmethod
    def h_inf(r_TC: torch.Tensor) -> torch.Tensor:
        """
        Courbe d'inactivation T-type à l'équilibre (Boltzmann).
        Proxy voltage V ≈ -70 + 0.5 × r_TC mV.
        h_inf → 1 quand TC hyperpolarisé (r_TC bas) → canal T prêt à burster.
        """
        V = -70.0 + 0.5 * r_TC
        return 1.0 / (1.0 + torch.exp((V + 65.0) / 5.0))

    # ── Compartiments dendritiques ───────────────────────────────────────────

    def dendritic_step(self, r: torch.Tensor, mu: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcul de l'erreur prédictive dendritique pour les pyramidaux L2/3.

        Basal V^b ← L4 (bottom-up) inhibé par PVb
        Apical V^a ← μ (top-down)  inhibé par SSTm

        Éq. 3 : C_m dV^b/dt = -g_L(V^b-E_L) + g_b4*r_4*(E_E-V^b) - g_bpv*r_pvb*(V^b-E_I)
        Éq. 4 : ε_s = φ_b(V^b) - φ_a(V^a)
        Éq. 5 : r_s = φ_s(V^b + λ P_Ca V^a)

        Invariant I4 : P_Ca ∈ {0.0, 1.0}
        """
        r_L4   = r[IDX['L4']]
        r_PVb  = r[IDX['PVb']]
        r_SSTm = r[IDX['SSTm']]

        # Courants effectifs (normalisés, proxy mV via scaling 0.5)
        I_b = self.G_B4 * r_L4 - self.G_BPVB * r_PVb   # basal
        I_a = self.G_ATOP * mu - self.G_ASST * r_SSTm   # apical

        # Erreur prédictive locale (normalisée sur [−1, 1])
        eps_s = (I_b - I_a) / R_MAX

        # Plateau calcique : seuil sur l'erreur normalisée
        P_Ca = (eps_s.abs() > THETA_CA).float()

        # Taux L2/3 avec amplification burst (Éq. 5)
        r_s_new = self.phi(I_b + LAM_BURST * P_Ca * I_a)

        return eps_s, P_Ca, r_s_new

    # ── Pas de simulation ────────────────────────────────────────────────────

    def step(self, r: torch.Tensor, s: torch.Tensor,
             mu: torch.Tensor, u_ACh: float = 0.0
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Intégration Euler d'un pas dt pour les 14 variables.

        Args:
            r      : (14,) état courant
            s      : scalaire — entrée thalamique brute (Hz)
            mu     : scalaire — prédiction top-down (Hz)
            u_ACh  : scalaire — niveau acétylcholine [0, 1]

        Returns:
            r_new   : (14,) nouvel état
            P_Ca    : scalaire — plateau calcique (0 ou 1)
            eps_th  : scalaire — erreur thalamocorticale
            eps_s   : scalaire — erreur prédictive dendritique
        """
        # ── 1. Lésion L6 ─────────────────────────────────────────────────────
        if self.lesion_L6:
            r = r.clone()
            r[IDX['L6']] = 0.0

        r_L6  = r[IDX['L6']]
        r_TC  = r[IDX['TC']]
        h_T   = r[IDX['hT']]
        r_NRT = r[IDX['NRT']]

        # ── 2. Circuit thalamique (Éq. TC, hT, NRT) ──────────────────────────
        # Canal T-type : courant de rebond quand TC était hyperpolarisé
        # I_T > 0 uniquement si TC sous le repos (r_TC bas = hyperpolarisé)
        I_T = self.G_T * h_T * torch.clamp(20.0 - r_TC, min=0.0) / R_MAX

        # Relais thalamique : reçoit s, inhibé par NRT, modulé par L6 et I_T
        dr_TC = (-r_TC + self.phi(
            s - self.W_NRT_TC * r_NRT + self.W_TC6 * r_L6 + I_T * R_MAX,
            beta=0.12
        )) / TAU_MS['TC']

        # Inactivation T-type : lente (τ=100 ms), monte quand TC hyperpolarisé
        dh_T = (self.h_inf(r_TC) - h_T) / TAU_MS['hT']

        # NRT : activé par L6 (fort) et par TC (réciprocité)
        dr_NRT = (-r_NRT + self.phi(
            self.W_NRT_6 * r_L6 + self.W_TC_NRT * r_TC,
            beta=0.15
        )) / TAU_MS['NRT']

        # ── 3. Drive vers L4 depuis TC filtré ────────────────────────────────
        # L4 reçoit le relais thalamique (pas s directement)
        s_cortex = r_TC

        # ── 4. Dynamique des 11 populations corticales (Éq. 1) ───────────────
        # Injecter le poids appris dans J
        self.J[IDX['Ls'], IDX['L4']] = self.J_s4.clamp(0.0, 3.0)

        # Input externe par population
        b_thal = torch.zeros(NC)
        b_thal[IDX['L4']]    = 1.00  # L4 reçoit TC
        b_thal[IDX['PVb']]   = 0.30  # PVb aussi (inhibition feedforward rapide)
        b_thal[IDX['PVc']]   = 0.20
        b_thal[IDX['SSTnm']] = 0.10

        b_top = torch.zeros(NC)
        b_top[IDX['Ls']]    = 0.20   # top-down direct sur L2/3
        b_top[IDX['Ld']]    = 0.15   # top-down sur L5
        b_top[IDX['VIP1']]  = 0.40   # top-down ouvre VIP1 → désinhibition
        b_top[IDX['VIP2']]  = 0.10
        b_top[IDX['NGF']]   = 0.08

        b_ach = torch.zeros(NC)
        b_ach[IDX['VIP1']]  = 0.60   # ACh → VIP1 fort
        b_ach[IDX['VIP2']]  = 0.30

        net = self.J @ r[:NC] + b_thal * s_cortex + b_top * mu + b_ach * u_ACh

        dr_cortex = (-r[:NC] + self.phi(net, beta=0.10)) / TAU[:NC]

        # ── 5. Compartiments dendritiques et correction burst ─────────────────
        eps_s, P_Ca, r_s_burst = self.dendritic_step(r, mu)
        # Remplacer la dynamique L2/3 par la version dendritique
        dr_cortex[IDX['Ls']] = (r_s_burst - r[IDX['Ls']]) / TAU_MS['Ls']

        # ── 6. Intégration Euler ──────────────────────────────────────────────
        r_new = r.clone()
        r_new[:NC]          += self.dt * dr_cortex
        r_new[IDX['TC']]    += self.dt * dr_TC
        r_new[IDX['hT']]    += self.dt * dh_T
        r_new[IDX['NRT']]   += self.dt * dr_NRT

        # Clamps
        r_new[:NC]           = r_new[:NC].clamp(0.0, R_MAX)
        r_new[IDX['TC']]     = r_new[IDX['TC']].clamp(0.0, R_MAX)
        r_new[IDX['hT']]     = r_new[IDX['hT']].clamp(0.0, 1.0)
        r_new[IDX['NRT']]    = r_new[IDX['NRT']].clamp(0.0, R_MAX)

        # ── 7. Erreur thalamocorticale (Éq. 6) ───────────────────────────────
        eps_th = s - self.W_TC6 * r_L6

        # ── 8. STDP tri-factorielle (Éq. dJ/dt) ──────────────────────────────
        self._stdp_step(r_new, P_Ca)

        return r_new, P_Ca, eps_th, eps_s

    def _stdp_step(self, r: torch.Tensor, P_Ca: torch.Tensor) -> None:
        """
        Règle tri-factorielle (Mikulasch 2022) :
          ΔW = η · STDP(Δt) · P_Ca · (1 + 2·r_VIP1/R_MAX) − η_dec · W

        Implémentation par traces exponentielles.
        Invariant I5 : J_s4_learned ≥ 0
        """
        r_4    = r[IDX['L4']]   / R_MAX  # normalisé [0,1]
        r_s    = r[IDX['Ls']]   / R_MAX
        r_vip1 = r[IDX['VIP1']] / R_MAX

        decay = math.exp(-self.dt / self.TAU_STDP)
        self.tr_pre  = self.tr_pre  * decay + r_4
        self.tr_post = self.tr_post * decay + r_s

        # STDP symétrique : pré avant post → LTP, post avant pré → LTD
        stdp = self.tr_pre * r_s - 0.5 * self.tr_post * r_4

        gate = P_Ca * (1.0 + 2.0 * r_vip1)

        with torch.no_grad():
            self.J_s4 = (self.J_s4
                         + self.ETA * stdp * gate
                         - self.ETA_DEC * self.J_s4
                         ).clamp(0.0, 3.0)

    def apply_lesion(self, active: bool = True) -> None:
        """Active ou désactive la lésion L6."""
        self.lesion_L6 = active
        print(f"Lésion L6 : {'ON' if active else 'OFF'}")


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    T_total:    float = 2000.0,   # ms
    T_lesion:   float = 1000.0,   # ms (−1 = jamais)
    s_amp:      float = 30.0,     # Hz amplitude signal thalamique
    mu_amp:     float = 20.0,     # Hz amplitude prédiction top-down
    dt:         float = 0.1,      # ms
    seed:       int   = 42,
) -> dict:
    """
    Phase 1 [0, T_lesion]      : circuit intact (gamma dominant)
    Phase 2 [T_lesion, T_total]: lésion L6 (bascule vers bouffées bêta)

    Returns dict de traces numpy.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model   = MiniColumnV3(dt=dt)
    n_steps = int(T_total / dt)
    t_arr   = np.linspace(0.0, T_total, n_steps, endpoint=False)

    # Pré-allocation
    traces    = np.zeros((n_steps, N),  dtype=np.float32)
    pca_arr   = np.zeros(n_steps,       dtype=np.float32)
    epsth_arr = np.zeros(n_steps,       dtype=np.float32)
    epss_arr  = np.zeros(n_steps,       dtype=np.float32)
    js4_arr   = np.zeros(n_steps,       dtype=np.float32)

    # État initial
    r = torch.zeros(N)
    r[IDX['TC']]   = 15.0   # thalamus actif au repos
    r[IDX['hT']]   = 0.40   # inactivation T-type partielle
    r[IDX['L4']]   = 5.0
    r[IDX['NRT']]  = 5.0

    les_idx = int(T_lesion / dt) if T_lesion >= 0 else n_steps + 1
    rng = np.random.default_rng(seed)

    print(f"\n{'═'*60}")
    print(f"Simulation v3 : T={T_total:.0f} ms, dt={dt} ms, {n_steps:,} pas")
    print(f"Lésion L6 à   : t={T_lesion:.0f} ms")
    print(f"{'═'*60}")

    for i in range(n_steps):
        ti = t_arr[i]

        # Entrée thalamique : sinusoïde lente + bruit
        s  = float(np.clip(
            s_amp  * (0.5 + 0.5 * math.sin(2 * math.pi * ti / 250.0))
            + rng.standard_normal() * 4.0,
            0.0, R_MAX
        ))
        # Prédiction top-down : légèrement décalée
        mu = float(np.clip(
            mu_amp * (0.5 + 0.5 * math.sin(2 * math.pi * ti / 250.0 - math.pi / 5))
            + rng.standard_normal() * 2.0,
            0.0, R_MAX
        ))
        # ACh pulsée toutes les 600 ms
        u_ACh = 0.4 if (int(ti / 600) % 2 == 0) else 0.0

        # Lésion L6
        if i == les_idx:
            model.apply_lesion(True)

        # Pas de simulation
        r, P_Ca, eps_th, eps_s = model.step(
            r, torch.tensor(s), torch.tensor(mu), u_ACh
        )

        traces[i]    = r.numpy()
        pca_arr[i]   = float(P_Ca)
        epsth_arr[i] = float(eps_th)
        epss_arr[i]  = float(eps_s)
        js4_arr[i]   = float(model.J_s4)

        if i % 20000 == 0:
            print(f"  t={ti:6.0f} ms | r_Ls={r[IDX['Ls']]:5.1f} Hz | "
                  f"r_Ld={r[IDX['Ld']]:5.1f} Hz | "
                  f"r_TC={r[IDX['TC']]:5.1f} Hz | "
                  f"h_T={r[IDX['hT']]:.3f} | "
                  f"P_Ca={float(P_Ca):.0f} | J_s4={float(model.J_s4):.3f}")

    print(f"{'═'*60}\n")
    return {
        't'        : t_arr,
        'traces'   : traces,
        'P_Ca'     : pca_arr,
        'eps_th'   : epsth_arr,
        'eps_s'    : epss_arr,
        'J_s4'     : js4_arr,
        'T_lesion' : T_lesion,
        'dt'       : dt,
        'les_idx'  : les_idx,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE SPECTRALE ET VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_psd(signal: np.ndarray, dt_ms: float,
                f_max: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    """PSD par FFT sur un segment centré."""
    sig = signal - signal.mean()
    sig *= np.hanning(len(sig))
    n   = len(sig)
    fs  = 1000.0 / dt_ms          # fréquence d'échantillonnage en Hz
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd   = (np.abs(np.fft.rfft(sig)) ** 2) / n
    mask  = freqs <= f_max
    return freqs[mask], psd[mask]


def plot_results(res: dict, save_path: Optional[str] = None) -> None:
    """
    Figure à 5 panneaux :
      1. r_Ls (gamma FF) et r_Ld (bêta FB) — taux de population
      2. Circuit thalamique : r_TC, h_T, r_NRT
      3. PSD avant / après lésion — basculement gamma→bêta
      4. Inhibiteurs : PVb, SSTm, VIP1
      5. Plasticité : J_s4 et P_Ca
    """
    t         = res['t']
    traces    = res['traces']
    les_idx   = res['les_idx']
    dt        = res['dt']
    T_les     = res['T_lesion']

    r_Ls  = traces[:, IDX['Ls']]
    r_Ld  = traces[:, IDX['Ld']]
    r_TC  = traces[:, IDX['TC']]
    h_T   = traces[:, IDX['hT']]
    r_NRT = traces[:, IDX['NRT']]
    r_PVb = traces[:, IDX['PVb']]
    r_SSTm= traces[:, IDX['SSTm']]
    r_VIP1= traces[:, IDX['VIP1']]

    fig = plt.figure(figsize=(16, 20))
    gs  = gridspec.GridSpec(5, 1, hspace=0.45)
    axes = [fig.add_subplot(gs[k]) for k in range(5)]

    BLUE  = '#1565C0'
    RED   = '#C62828'
    GREY  = '#757575'
    VERT  = '#2E7D32'

    def vline(ax):
        ax.axvline(T_les, color=GREY, lw=1.8, ls='--', alpha=0.8)

    # ── Panneau 1 : r_Ls et r_Ld ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, r_Ls, color=BLUE,  lw=0.6, alpha=0.85, label='r_Ls L2/3 (FF gamma)')
    ax.plot(t, r_Ld, color='darkorange', lw=0.6, alpha=0.85, label='r_Ld L5 (FB bêta)')
    vline(ax)
    ax.set_ylabel('Taux (Hz)', fontsize=10)
    ax.set_title('L2/3 (gamma) et L5 (bêta) — lésion L6 au trait gris', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, t[-1])

    # ── Panneau 2 : circuit thalamique ───────────────────────────────────────
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(t, r_TC,  color='purple', lw=0.7, label='r_TC relais')
    ax.plot(t, r_NRT, color='teal',   lw=0.7, alpha=0.9, label='r_NRT réticulaire')
    ax2.plot(t, h_T,  color='brown',  lw=1.0, ls='--', alpha=0.8, label='h_T canal T')
    vline(ax)
    ax.set_ylabel('Taux (Hz)',       fontsize=10, color='purple')
    ax2.set_ylabel('Inactivation T', fontsize=10, color='brown')
    ax2.set_ylim(0, 1)
    ax.set_title('Circuit thalamique — montée h_T post-lésion → bouffées bêta NRT', fontsize=11)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc='upper left')
    ax.set_xlim(0, t[-1])

    # ── Panneau 3 : PSD avant / après lésion ─────────────────────────────────
    ax = axes[2]
    w = min(les_idx, len(t) - les_idx)

    if w > 100:
        # L2/3
        f_pre,  p_ls_pre  = compute_psd(r_Ls[les_idx - w : les_idx],      dt)
        f_post, p_ls_post = compute_psd(r_Ls[les_idx     : les_idx + w],   dt)
        # L5
        _,      p_ld_pre  = compute_psd(r_Ld[les_idx - w : les_idx],      dt)
        _,      p_ld_post = compute_psd(r_Ld[les_idx     : les_idx + w],   dt)

        ax.semilogy(f_pre,  p_ls_pre,  color=BLUE,  lw=1.8, label='r_Ls avant')
        ax.semilogy(f_post, p_ls_post, color=RED,   lw=1.8, label='r_Ls après')
        ax.semilogy(f_pre,  p_ld_pre,  color='steelblue', lw=1.0, ls='--',
                    alpha=0.7, label='r_Ld avant')
        ax.semilogy(f_post, p_ld_post, color='salmon',    lw=1.0, ls='--',
                    alpha=0.7, label='r_Ld après')

    ax.axvspan(40,  80, alpha=0.10, color='blue',   label='gamma (40–80 Hz)')
    ax.axvspan(12,  20, alpha=0.10, color='orange', label='bêta  (12–20 Hz)')
    ax.set_xlabel('Fréquence (Hz)', fontsize=10)
    ax.set_ylabel('PSD (log)',      fontsize=10)
    ax.set_title('Spectre avant / après lésion L6 — basculement gamma → bêta', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 120)

    # ── Panneau 4 : inhibiteurs ───────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, r_PVb,  color='crimson',    lw=0.6, alpha=0.9, label='r_PVb basket (gain)')
    ax.plot(t, r_SSTm, color='darkcyan',   lw=0.6, alpha=0.9, label='r_SSTm Martinotti (apical)')
    ax.plot(t, r_VIP1, color='goldenrod',  lw=0.6, alpha=0.9, label='r_VIP1 (désinhibition)')
    vline(ax)
    ax.set_ylabel('Taux (Hz)', fontsize=10)
    ax.set_title('Interneurones — PVb, SSTm, VIP1', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, t[-1])

    # ── Panneau 5 : plasticité ────────────────────────────────────────────────
    ax = axes[4]
    ax2 = ax.twinx()
    ax.plot(t, res['J_s4'], color=VERT,  lw=1.2, label='J_s4 (appris)')
    ax2.fill_between(t, res['P_Ca'], alpha=0.25, color=RED, label='P_Ca')
    vline(ax)
    ax.set_ylabel('Poids J_{s4}',         fontsize=10, color=VERT)
    ax2.set_ylabel('P_Ca plateau Ca',     fontsize=10, color=RED)
    ax2.set_ylim(0, 2)
    ax.set_title('Plasticité tri-factorielle — J_s4 L4→L2/3 + plateau calcique', fontsize=11)
    ax.set_xlabel('Temps (ms)', fontsize=10)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=9)
    ax.set_xlim(0, t[-1])

    plt.suptitle('MiniColumn v3 — 11 populations + TC/NRT + STDP tri-factorielle',
                 fontsize=13, fontweight='bold', y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvée → {save_path}")
    else:
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STATISTIQUES RAPIDES
# ══════════════════════════════════════════════════════════════════════════════

def print_stats(res: dict) -> None:
    les = res['les_idx']
    t   = res['traces']
    r_Ls = t[:, IDX['Ls']]
    r_Ld = t[:, IDX['Ld']]
    r_TC = t[:, IDX['TC']]
    h_T  = t[:, IDX['hT']]
    dt   = res['dt']

    def band_power(signal, start, end, f_lo, f_hi):
        freqs, psd = compute_psd(signal[start:end], dt)
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        return float(psd[mask].sum()) if mask.any() else 0.0

    w = min(les, len(r_Ls) - les)
    print("\n── Statistiques de simulation ──────────────────────────────")
    print(f"  r_Ls avant  : {r_Ls[:les].mean():5.1f} ± {r_Ls[:les].std():4.1f} Hz")
    print(f"  r_Ls après  : {r_Ls[les:].mean():5.1f} ± {r_Ls[les:].std():4.1f} Hz")
    print(f"  r_Ld avant  : {r_Ld[:les].mean():5.1f} ± {r_Ld[:les].std():4.1f} Hz")
    print(f"  r_Ld après  : {r_Ld[les:].mean():5.1f} ± {r_Ld[les:].std():4.1f} Hz")
    print(f"  h_T avant   : {h_T[:les].mean():.3f}  →  après : {h_T[les:].mean():.3f} "
          f"(montée post-lésion attendue)")
    if w > 100:
        g_pre  = band_power(r_Ls, les - w, les, 40, 80)
        g_post = band_power(r_Ls, les,     les + w, 40, 80)
        b_pre  = band_power(r_Ld, les - w, les, 12, 20)
        b_post = band_power(r_Ld, les,     les + w, 12, 20)
        print(f"\n  Puissance gamma r_Ls : {g_pre:.1f} → {g_post:.1f}  "
              f"({'↓' if g_post < g_pre else '↑'})")
        print(f"  Puissance bêta  r_Ld : {b_pre:.1f} → {b_post:.1f}  "
              f"({'↑' if b_post > b_pre else '↓'})")
    print(f"\n  J_s4 initial : 1.500")
    print(f"  J_s4 final   : {res['J_s4'][-1]:.3f}")
    print(f"  P_Ca moy. avant : {res['P_Ca'][:les].mean():.3f}")
    print(f"  P_Ca moy. après : {res['P_Ca'][les:].mean():.3f}")
    print("────────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os

    results = run_simulation(
        T_total  = 2000.0,
        T_lesion = 1000.0,
        s_amp    = 30.0,
        mu_amp   = 20.0,
        dt       = 0.1,
        seed     = 42,
    )

    print_stats(results)

    # Sauvegarde figure dans le même dossier que le script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path   = os.path.join(script_dir, 'minicolumn_v3_simulation.png')
    plot_results(results, save_path=fig_path)
