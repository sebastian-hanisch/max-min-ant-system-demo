"""Konstanten der Max-Min-Ant-System-Demo: Vehikel (wie ant-system-demo, diskrete Lieferroute), MMAS-Regler,
Presets (Presets folgen nach den Messungen)."""

# --- Vehikel: Lieferroute (wortgleich aus genetic-algorithm-demo/ga_constants.py, wie ant-system-demo) -------------------------------------

AREA = 100.0
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0                # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0              # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
N_MIN, N_MAX, DEFAULT_N, N_STEP = 8, 100, 30, 2    # N_MIN=8, damit der Slider auch die kleine Vergleichsinstanz abbilden kann (wie ant-system-demo)

# --- Max-Min Ant System --------------------------------------------------------------------------------------------------------------------
# Übergangswahrscheinlichkeit/Tourkonstruktion identisch zu Ant System - der Unterschied liegt im Pheromon-Update:
# nur die beste Ameise der Generation legt ab, Pheromon ist auf [tau_min, tau_max] begrenzt.

ANTS_MIN, ANTS_MAX, DEFAULT_ANTS, ANTS_STEP = 4, 100, 20, 2
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 300, 80, 10
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA, ALPHA_STEP = 0.0, 5.0, 1.0, 0.1     # Pheromon-Exponent
BETA_MIN, BETA_MAX, DEFAULT_BETA, BETA_STEP = 0.0, 8.0, 3.0, 0.5        # Sichtbarkeits-Exponent (1/Distanz)
RHO_MIN, RHO_MAX, DEFAULT_RHO, RHO_STEP = 0.05, 0.95, 0.3, 0.05         # Verdunstungsrate (kleiner Standard als bei Ant System - MMAS' Grenzen puffern gegen zu schnelle Verdunstung)
RATIO_MIN, RATIO_MAX, DEFAULT_RATIO, RATIO_STEP = 0.001, 0.5, 0.05, 0.005   # tau_min = tau_max * RATIO - eigener Regler
Q = 1.0                            # Ablage-Konstante (Δτ = Q / Tourlänge)
SEED_MAX = 999999
DEFAULT_SEED = 35                  # Vehikel-Seed (wie ant-system-demo/genetic-algorithm-demo)
DEFAULT_RUN_SEED = 7               # Seed des MMAS-Laufs selbst

# --- Kopfexperiment: validiert den Fix, selbst gemessen (kein Cross-Repo-Zitat - andere Vergleichsgröße als ------------------------------------
# ant-system-demos Brute-Force-Gap) - Anteil der Pheromonwerte nahe tau_max nach einem LANGEN Lauf, bei lockerem
# gegen eng gewähltem Verhältnis r.

STAGNATION_RATIO_LOOSE = 0.02       # locker (kleines r -> tau_min nahe 0 -> weiter Abstand zu tau_max) - nähert sich unbeschränktem Ant-System-Verhalten an
STAGNATION_RATIO_TIGHT = 0.5        # eng (großes r -> tau_min nahe tau_max) - MMAS' übliche Größenordnung ist eher niedrig, dies ist absichtlich das enge Gegenstück
STAGNATION_NEAR_MAX_THRESHOLD = 0.9  # ein Pheromonwert gilt als "nahe tau_max", wenn er über diesem Anteil von tau_max liegt
COMPARISON_N = 8
COMPARISON_VEHICLE_SEED = 19
COMPARISON_SEEDS = tuple(range(2800000, 2800020))
COMPARISON_ANTS, COMPARISON_GENS = 20, 80
STAGNATION_GENS = 150                # langer Lauf für das Stagnations-Kopfexperiment

# --- Eigener Regler: tau-Verhältnis r --------------------------------------------------------------------------------------------------------

RATIO_VALUES = (0.005, 0.02, 0.05, 0.15, 0.4)
RATIO_EXPERIMENT_SEEDS = tuple(range(2900000, 2900020))
RATIO_EXPERIMENT_ANTS = 10
RATIO_EXPERIMENT_GENS = 30

SWEEP_SEEDS = tuple(range(3000000, 3000005))
SWEEP_VALUES = {"ants": (4, 10, 20, 40, 80), "ratio": RATIO_VALUES}
SWEEP_LABELS = {"ants": "Ameisenzahl", "ratio": "τ-Verhältnis r (τ_min/τ_max)"}


def _preset(n=DEFAULT_N, ants=DEFAULT_ANTS, gens=DEFAULT_GEN, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, rho=DEFAULT_RHO, ratio=DEFAULT_RATIO, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    return {"n": n, "ants": ants, "gens": gens, "alpha": alpha, "beta": beta, "rho": rho, "ratio": ratio, "seed": seed, "run_seed": run_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Enges τ-Verhältnis": _preset(ratio=RATIO_VALUES[0]),
    "Lockeres τ-Verhältnis": _preset(ratio=RATIO_VALUES[-1]),
    "Kleine Instanz (Vergleich mit Brute-Force)": _preset(n=COMPARISON_N, seed=COMPARISON_VEHICLE_SEED, ants=COMPARISON_ANTS, gens=COMPARISON_GENS),
}
PRESET_HELP = {
    "Standardfall": "30 Stopps, 20 Ameisen, 80 Generationen, τ-Verhältnis r=0.05 - findet 491.1 km.",
    "Enges τ-Verhältnis": "r=0.005 (τ_min sehr nahe τ_max) - auf diesem Vehikel identisch zum Standardfall (491.1 km); der Effekt zeigt sich erst im eigenen Regler-Experiment über 20 Läufe.",
    "Lockeres τ-Verhältnis": "r=0.4 (τ_min weit unter τ_max, näher am unbeschränkten Ant-System-Verhalten) - 520.5 km, spürbar schlechter als der Standardfall.",
    "Kleine Instanz (Vergleich mit Brute-Force)": "8 Stopps - MMAS trifft mit 255.4 km exakt das Brute-Force-Optimum (0.0 % Abstand).",
}
