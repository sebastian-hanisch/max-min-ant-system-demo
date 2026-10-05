"""Orakel-Tests (unabhängiger Rechenweg): (1) jede Generation eines MMAS-Laufs wird aus den gespeicherten Touren mit
einer Schleifenfassung des Updates nachgerechnet (nur die beste Tour der Generation legt ab, τ_max = 1/(ρ·L_best),
τ_min = r·τ_max, Klemmung nur auf echten Kanten); (2) die Brute-Force-Referenz wird gegen Held-Karp-DP geprüft."""

import numpy as np

import mmas_algorithm as A
import mmas_evaluation as E


def _tour_len(t, D):
    return sum(D[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))


def _nn_len(D):
    n = len(D)
    cur, seen, tot = 0, {0}, 0.0
    while len(seen) < n:
        nxt = min((j for j in range(n) if j not in seen), key=lambda j: D[cur][j])
        tot += D[cur][nxt]
        seen.add(nxt)
        cur = nxt
    return tot + D[cur][0]


def test_every_generation_matches_loop_replay():
    rng = np.random.default_rng(2026)
    for k in range(25):
        n = int(rng.integers(5, 10))
        ants = int(rng.integers(2, 9))
        gens = int(rng.integers(2, 8))
        rho = float(rng.choice([0.05, 0.3, 0.95]))
        ratio = float(rng.choice([0.001, 0.05, 0.5]))
        D = A.dist_matrix(rng.random((n, 2)) * 100)
        r = A.run_mmas(D, ants, gens, 1.0, 3.0, rho, 1.0, ratio, k, keep_history=True)
        t_max = 1 / (rho * _nn_len(D))
        t_min = t_max * ratio
        tau_prev = np.full((n, n), t_max)
        best = float("inf")
        for gen in r.generations:
            lens = [_tour_len(list(t), D) for t in gen.tours]
            gi = int(np.argmin(lens))
            if lens[gi] < best:
                best = lens[gi]
                t_max = 1 / (rho * best)
                t_min = t_max * ratio
            t = list(gen.tours[gi])
            edges = {(t[i], t[(i + 1) % n]) for i in range(n)} | {(t[(i + 1) % n], t[i]) for i in range(n)}
            for i in range(n):
                for j in range(n):
                    if i == j:
                        assert gen.tau[i, j] == 0.0
                        continue
                    v = (1 - rho) * tau_prev[i, j] + (1.0 / lens[gi] if (i, j) in edges else 0.0)
                    assert abs(gen.tau[i, j] - min(max(v, t_min), t_max)) < 1e-12
            assert abs(gen.t_max - t_max) < 1e-12 * t_max
            tau_prev = gen.tau


def _held_karp(D):
    n = len(D)
    dp = {(1 << j, j): D[0, j] for j in range(1, n)}
    for mask in range(2, 1 << n, 2):
        for j in range(1, n):
            if (mask, j) not in dp:
                continue
            for k in range(1, n):
                if not mask >> k & 1:
                    key = (mask | 1 << k, k)
                    dp[key] = min(dp.get(key, 1e18), dp[(mask, j)] + D[j, k])
    full = sum(1 << j for j in range(1, n))
    return min(dp[(full, j)] + D[j, 0] for j in range(1, n))


def test_brute_force_reference_matches_held_karp():
    rng = np.random.default_rng(7)
    for n in (4, 5, 6, 7):
        for _ in range(3):
            D = A.dist_matrix(rng.random((n, 2)) * 100)
            assert abs(E.brute_force(D) - _held_karp(D)) < 1e-9
