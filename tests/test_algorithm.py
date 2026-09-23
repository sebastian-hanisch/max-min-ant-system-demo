"""Handrechnungen für τ-Grenzen, Best-Ant-Update mit Klemmung, Übergangswahrscheinlichkeit und die
Nächster-Nachbar-Schätzung, Brute-Force-Vergleich auf einer sehr kleinen Instanz."""

from itertools import permutations

import numpy as np
import pytest

import mmas_algorithm as A


def test_tau_bounds_matches_hand_calculation():
    t_min, t_max = A.tau_bounds(rho=0.5, best_length=10.0, ratio=0.05)
    assert t_max == pytest.approx(0.2)     # 1/(0.5*10)
    assert t_min == pytest.approx(0.01)    # 0.2*0.05


def test_mmas_update_matches_hand_calculation_without_clipping():
    tau = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    new_tau = A.mmas_update(tau, best_tour=[0, 1, 2], best_length=10.0, rho=0.5, q=1.0, t_min=0.01, t_max=0.7)
    expected = np.array([[0.0, 0.6, 0.6], [0.6, 0.0, 0.6], [0.6, 0.6, 0.0]])
    assert new_tau == pytest.approx(expected)


def test_mmas_update_clips_to_t_max():
    tau = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    new_tau = A.mmas_update(tau, best_tour=[0, 1, 2], best_length=10.0, rho=0.5, q=1.0, t_min=0.01, t_max=0.55)
    # ohne Klemmung waeren es 0.6 (siehe Test oben) - mit t_max=0.55 muss geklemmt werden
    assert new_tau[0, 1] == pytest.approx(0.55)
    assert np.all(new_tau <= 0.55 + 1e-12)


def test_mmas_update_never_puts_pheromone_on_the_diagonal():
    """Regressionstest: Klemmung auf t_min darf die Diagonale (Selbstschleifen, keine echten Kanten) nicht
    faelschlich auf t_min > 0 anheben."""
    tau = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    new_tau = A.mmas_update(tau, best_tour=[0, 1, 2], best_length=10.0, rho=0.5, q=1.0, t_min=0.01, t_max=0.7)
    assert np.all(np.diag(new_tau) == 0.0)


def test_mmas_update_clips_to_t_min():
    tau = np.array([[0.0, 0.001, 0.001], [0.001, 0.0, 0.001], [0.001, 0.001, 0.0]])
    # (1-0.99)*0.001 + 1/1000 = 0.00001 + 0.001 = 0.00101, auf jeder der 3 Kanten (ein Dreieck hat nur 3 Kanten,
    # die Tour 0-1-2-0 nutzt sie alle) - liegt unter t_min=0.01 und muss dort hin geklemmt werden
    new_tau = A.mmas_update(tau, best_tour=[0, 1, 2], best_length=1000.0, rho=0.99, q=1.0, t_min=0.01, t_max=1.0)
    assert new_tau[1, 2] == pytest.approx(0.01)


def test_transition_probs_matches_hand_calculation():
    tau = np.array([[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    eta = np.array([[0.0, 0.5, 0.25], [0.5, 0.0, 1.0], [0.25, 1.0, 0.0]])
    probs = A.transition_probs(0, np.array([1, 2]), tau, eta, alpha=1, beta=1)
    assert probs == pytest.approx([1.0 / 1.75, 0.75 / 1.75])


def test_construct_tour_visits_every_node_exactly_once():
    rng = np.random.default_rng(1)
    n = 6
    tau = np.ones((n, n))
    eta = np.ones((n, n))
    np.fill_diagonal(tau, 0.0)
    np.fill_diagonal(eta, 0.0)
    for start in range(n):
        tour = A.construct_tour(tau, eta, alpha=1.0, beta=1.0, rng=rng, start=start)
        assert sorted(tour.tolist()) == list(range(n))
        assert tour[0] == start


def test_nearest_neighbor_tour_length_matches_manual_computation():
    # 4 Punkte auf einer Linie: 0, 1, 3, 6 - vom Start 0 aus ist die gierige Nächster-Nachbar-Tour 0-1-3-6-0
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    D = A.dist_matrix(xy)
    length = A.nearest_neighbor_tour_length(D, start=0)
    assert length == pytest.approx(1.0 + 2.0 + 3.0 + 6.0)    # 0->1->3->6->0


# --- Brute-Force-Vergleich auf einer sehr kleinen Instanz -------------------------------------------------------------------------------------


def brute_force_optimum(D):
    n = D.shape[0]
    best = None
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        length = A.tour_length(tour, D)
        if best is None or length < best:
            best = length
    return best


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_mmas_finds_the_brute_force_optimum_on_a_tiny_instance(seed):
    rng = np.random.default_rng(seed)
    n = 6
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    optimum = brute_force_optimum(D)

    r = A.run_mmas(D, n_ants=15, generations=40, alpha=1.0, beta=3.0, rho=0.3, q=1.0, ratio=0.05, seed=seed)
    assert r.best_length == pytest.approx(optimum, rel=0.05)


def test_run_mmas_tau_always_within_bounds():
    rng = np.random.default_rng(1)
    n = 8
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    r = A.run_mmas(D, n_ants=10, generations=20, alpha=1.0, beta=3.0, rho=0.3, q=1.0, ratio=0.05, seed=1, keep_history=True)
    for g in r.generations:
        off_diag = g.tau[~np.eye(n, dtype=bool)]
        assert np.all(off_diag >= g.t_min - 1e-9)
        assert np.all(off_diag <= g.t_max + 1e-9)


def test_run_mmas_history_shapes_and_generations_snapshots():
    rng = np.random.default_rng(1)
    n = 6
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    gens = 10
    r = A.run_mmas(D, n_ants=8, generations=gens, alpha=1.0, beta=3.0, rho=0.3, q=1.0, ratio=0.05, seed=1, keep_history=True)
    assert r.best_history.shape == (gens,)
    assert len(r.generations) == gens
    for g in r.generations:
        assert g.tau.shape == (n, n)
        assert g.tours.shape == (8, n)
        assert g.lengths.shape == (8,)


def test_run_mmas_best_history_is_monotonically_non_increasing():
    rng = np.random.default_rng(1)
    n = 8
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    r = A.run_mmas(D, n_ants=10, generations=30, alpha=1.0, beta=3.0, rho=0.3, q=1.0, ratio=0.05, seed=1)
    assert np.all(np.diff(r.best_history) <= 1e-9)


def test_run_mmas_without_history_leaves_generations_empty():
    rng = np.random.default_rng(1)
    xy = rng.random((5, 2)) * 100
    D = A.dist_matrix(xy)
    r = A.run_mmas(D, n_ants=5, generations=5, alpha=1.0, beta=3.0, rho=0.3, q=1.0, ratio=0.05, seed=1, keep_history=False)
    assert r.generations == []
