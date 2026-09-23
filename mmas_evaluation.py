"""Auswertung der Max-Min-Ant-System-Demo: ein Lauf, Sweep über Ameisenzahl/τ-Verhältnis, und zwei Experimente -
Kopfexperiment (validiert den Fix: Anteil der Pheromonwerte nahe τ_max nach einem langen Lauf, eng gegen locker
gewähltes τ-Verhältnis - selbst gemessen, kein Cross-Repo-Zitat) und eigener Regler (τ-Verhältnis r)."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

import numpy as np

import mmas_algorithm as A
import mmas_constants as C
import mmas_scenario as S

BRUTE_FORCE_MAX_N = 9      # (n-1)! Touren; bei 9 sind das 40 320 - noch < 1 s


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    ants: int = C.DEFAULT_ANTS
    gens: int = C.DEFAULT_GEN
    alpha: float = C.DEFAULT_ALPHA
    beta: float = C.DEFAULT_BETA
    rho: float = C.DEFAULT_RHO
    ratio: float = C.DEFAULT_RATIO
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=256)
def instance(n, seed):
    inst = S.generate_perm(n, 0, seed)
    return inst, A.dist_matrix(inst.xy)


def run(settings, keep_history=False):
    inst, D = instance(settings.n, settings.seed)
    return A.run_mmas(D, settings.ants, settings.gens, settings.alpha, settings.beta, settings.rho, C.Q, settings.ratio, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object
    inst: object
    D: np.ndarray
    brute_force_optimum: float             # nur bei n <= BRUTE_FORCE_MAX_N berechnet, sonst NaN

    @property
    def gap(self):
        if not np.isfinite(self.brute_force_optimum) or self.brute_force_optimum == 0:
            return float("nan")
        return max(0.0, 100.0 * (self.result.best_length - self.brute_force_optimum) / self.brute_force_optimum)


def brute_force(D):
    n = D.shape[0]
    best = None
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        length = A.tour_length(tour, D)
        if best is None or length < best:
            best = length
    return best


def analyse(settings, keep_history=True):
    result = run(settings, keep_history=keep_history)
    inst, D = instance(settings.n, settings.seed)
    optimum = brute_force(D) if settings.n + 1 <= BRUTE_FORCE_MAX_N else float("nan")
    return Analysis(settings, result, inst, D, optimum)


# --- Sweep (wie ant-system-demo) --------------------------------------------------------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    gaps = []
    for run_seed in seeds:
        a = analyse(replace(s0, run_seed=run_seed), keep_history=False)
        gaps.append(a.gap)
    return {"gap": float(np.nanmean(gaps))}


def sweep(param, base=None, values=None):
    # eigenes knappes Budget auf der kleinen Vergleichsinstanz - beim komfortablen Kopfexperiment-Budget saettigt
    # der Abstand zum Optimum fast überall bei ~0 %, wie bei ant-system-demo
    base = Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, ants=4, gens=8) if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]


# --- Experiment 1: Kopfexperiment - validiert den Fix (selbst gemessen) ------------------------------------------------------------------------


def stagnation_experiment(n=None, seed=None, ants=None, gens=None, ratio_loose=None, ratio_tight=None):
    """Anteil der Pheromonwerte, die nach einem LANGEN Lauf nahe τ_max "kleben" (ein Stagnations-Symptom) - eng
    gegen locker gewähltes τ-Verhältnis r. Kein Cross-Repo-Zitat (andere Vergleichsgröße als ant-system-demos
    Brute-Force-Gap), sondern MMAS' eigene Kernbehauptung direkt gemessen."""
    n = C.DEFAULT_N if n is None else n
    seed = C.DEFAULT_SEED if seed is None else seed
    ants = C.DEFAULT_ANTS if ants is None else ants
    gens = C.STAGNATION_GENS if gens is None else gens
    ratio_loose = C.STAGNATION_RATIO_LOOSE if ratio_loose is None else ratio_loose
    ratio_tight = C.STAGNATION_RATIO_TIGHT if ratio_tight is None else ratio_tight

    rows = {}
    for label, ratio in (("loose", ratio_loose), ("tight", ratio_tight)):
        s = Settings(n=n, seed=seed, ants=ants, gens=gens, ratio=ratio)
        r = run(s, keep_history=True)
        last_gen = r.generations[-1]
        n_nodes = last_gen.tau.shape[0]
        off_diag = last_gen.tau[~np.eye(n_nodes, dtype=bool)]
        near_max_share = float(np.mean(off_diag >= C.STAGNATION_NEAR_MAX_THRESHOLD * last_gen.t_max))
        rows[label] = {"near_max_share": near_max_share, "ratio": ratio}
    return rows


# --- Experiment 2: eigener Regler - tau-Verhältnis r --------------------------------------------------------------------------------------------


def ratio_experiment(n=None, seed=None, values=None, seeds=None, ants=None, gens=None):
    n = C.DEFAULT_N if n is None else n
    seed = C.DEFAULT_SEED if seed is None else seed
    values = C.RATIO_VALUES if values is None else values
    seeds = C.RATIO_EXPERIMENT_SEEDS if seeds is None else seeds
    ants = C.RATIO_EXPERIMENT_ANTS if ants is None else ants
    gens = C.RATIO_EXPERIMENT_GENS if gens is None else gens

    rows = []
    for ratio in values:
        lengths = []
        for run_seed in seeds:
            s = Settings(n=n, seed=seed, ants=ants, gens=gens, ratio=ratio, run_seed=run_seed)
            r = run(s, keep_history=False)
            lengths.append(r.best_length)
        rows.append({"ratio": ratio, "length_median": float(np.median(lengths))})
    return rows
