"""Max-Min-Ant-System-Kern (Stützle & Hoos, 1996/2000): behebt Ant Systems Stagnationsgefahr über zwei Änderungen -
Pheromonwerte auf [τ_min, τ_max] begrenzt, nur die beste Ameise einer Generation legt ab. Übergangswahrscheinlichkeit
und Tourkonstruktion sind inhaltlich identisch zu Ant System (kein Cross-Repo-Import, portfolioweite Konvention -
eigene Kernfunktionen). Bewusste Vereinfachungen (siehe README): τ_min = τ_max · r mit festem Verhältnis r statt
Stützle & Hoos' analytischer p_best-Herleitung; keine λ-Verzweigungsfaktor-Stagnationserkennung mit
Pheromon-Reinitialisierung; nur iteration-best (keine alternierende iteration-best/global-best-Ablage)."""

from dataclasses import dataclass, field

import numpy as np


def dist_matrix(xy):
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def tour_length(tour, D):
    idx = np.asarray(tour)
    return float(D[idx, np.roll(idx, -1)].sum())


def transition_probs(current, unvisited, tau, eta, alpha, beta):
    weights = (tau[current, unvisited] ** alpha) * (eta[current, unvisited] ** beta)
    total = weights.sum()
    if total <= 0:
        return np.full(len(unvisited), 1.0 / len(unvisited))
    return weights / total


def construct_tour(tau, eta, alpha, beta, rng, start):
    n = tau.shape[0]
    unvisited = [j for j in range(n) if j != start]
    tour = [start]
    current = start
    while unvisited:
        candidates = np.array(unvisited)
        probs = transition_probs(current, candidates, tau, eta, alpha, beta)
        next_node = int(rng.choice(candidates, p=probs))
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return np.array(tour)


def nearest_neighbor_tour_length(D, start=0):
    """Gierige Nächster-Nachbar-Tour - liefert eine erste Schätzung für τ_max, bevor irgendeine Ameise gelaufen ist
    (Standardpraxis: τ_max anfangs über eine Heuristik-Tour schätzen, siehe README)."""
    n = D.shape[0]
    unvisited = set(range(n)) - {start}
    tour = [start]
    current = start
    while unvisited:
        next_node = min(unvisited, key=lambda j: D[current, j])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return tour_length(tour, D)


def tau_bounds(rho, best_length, ratio):
    """τ_max = 1/(ρ·L_best) (Stützle & Hoos), τ_min = τ_max·r (vereinfachtes Verhältnis statt der analytischen
    p_best-Herleitung)."""
    t_max = 1.0 / (rho * best_length)
    return t_max * ratio, t_max


def mmas_update(tau, best_tour, best_length, rho, q, t_min, t_max):
    """τ <- (1-ρ)τ + Δτ_best (nur die Kanten der besten Tour DIESER Generation), dann auf [τ_min, τ_max] geklemmt.
    Die Diagonale (Selbstschleifen, keine echten Kanten) bleibt dabei immer bei 0 - die Klemmung auf t_min gilt
    nur für echte Kanten, sonst würde t_min > 0 fälschlich Pheromon auf Selbstschleifen erzeugen."""
    new_tau = (1.0 - rho) * tau
    contribution = q / best_length
    a = np.asarray(best_tour)
    b = np.roll(a, -1)
    new_tau[a, b] += contribution
    new_tau[b, a] += contribution
    new_tau = np.clip(new_tau, t_min, t_max)
    np.fill_diagonal(new_tau, 0.0)
    return new_tau


@dataclass
class Generation:
    tau: np.ndarray
    tours: np.ndarray
    lengths: np.ndarray
    best_tour: np.ndarray      # global bislang bestes
    t_min: float
    t_max: float


@dataclass
class MMASResult:
    best_tour: np.ndarray
    best_length: float
    best_history: np.ndarray       # (generations,) - bester bisher gefundener Wert NACH Generation g (1-indexiert)
    generations: list = field(default_factory=list)   # nur befüllt, wenn keep_history=True


def run_mmas(D, n_ants, generations, alpha, beta, rho, q, ratio, seed, keep_history=False):
    """Ein MMAS-Lauf. `D`: Distanzmatrix (n_nodes, n_nodes)."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    eta = np.zeros_like(D)
    mask = D > 0
    eta[mask] = 1.0 / D[mask]

    initial_estimate = nearest_neighbor_tour_length(D)
    t_min, t_max = tau_bounds(rho, initial_estimate, ratio)
    tau = np.full((n, n), t_max)
    np.fill_diagonal(tau, 0.0)

    best_tour = None
    best_length = float("inf")
    best_history = []
    gens_snapshots = []

    for _ in range(generations):
        starts = rng.integers(0, n, size=n_ants)
        tours = np.array([construct_tour(tau, eta, alpha, beta, rng, start=int(s)) for s in starts])
        lengths = np.array([tour_length(t, D) for t in tours])

        gen_best_idx = int(np.argmin(lengths))
        gen_best_tour = tours[gen_best_idx]
        gen_best_length = float(lengths[gen_best_idx])

        if gen_best_length < best_length:
            best_length = gen_best_length
            best_tour = gen_best_tour.copy()
            t_min, t_max = tau_bounds(rho, best_length, ratio)

        tau = mmas_update(tau, gen_best_tour, gen_best_length, rho, q, t_min, t_max)

        best_history.append(best_length)
        if keep_history:
            gens_snapshots.append(Generation(tau.copy(), tours.copy(), lengths.copy(), best_tour.copy(), t_min, t_max))

    return MMASResult(best_tour, best_length, np.array(best_history), gens_snapshots)
