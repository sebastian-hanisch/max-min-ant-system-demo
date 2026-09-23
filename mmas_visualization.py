"""Plotly-Abbildungen der Max-Min-Ant-System-Demo: Karte mit pheromonstärke-gewichteten Kanten (wachsendes
Beispiel), τ_min/τ_max-Verlauf, Vergleichs- und Sweep-Abbildungen. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import mmas_constants as C

STOP_COLOR = "#4c78a8"
BEST_COLOR = "#54a24b"
PHEROMONE_COLOR = "#e45756"
MMAS_COLOR = "#4c78a8"
LOOSE_COLOR = "#e45756"
TIGHT_COLOR = "#54a24b"
REF_COLOR = "#7f7f7f"

PHEROMONE_PERCENTILE = 80


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_pheromone_map(xy, generation, title=None):
    """Karte mit den Kanten mit der stärksten Pheromonkonzentration (oberhalb PHEROMONE_PERCENTILE) - Linienbreite/
    -deckkraft proportional zur Position zwischen τ_min und τ_max - plus der global besten Tour hervorgehoben."""
    fig = go.Figure()
    tau = generation.tau
    n = tau.shape[0]
    iu = np.triu_indices(n, 1)
    values = tau[iu]
    span = generation.t_max - generation.t_min
    if len(values) and span > 0:
        threshold = np.percentile(values, PHEROMONE_PERCENTILE)
        for a, b, v in zip(iu[0], iu[1], values):
            if v < threshold:
                continue
            frac = float(np.clip((v - generation.t_min) / span, 0.0, 1.0))
            fig.add_trace(go.Scatter(
                x=[xy[a, 0], xy[b, 0]], y=[xy[a, 1], xy[b, 1]], mode="lines",
                line=dict(color=PHEROMONE_COLOR, width=1 + 5 * frac), opacity=max(0.15, frac),
                showlegend=False, hoverinfo="skip",
            ))

    best_edges = _tour_edges_list(generation.best_tour)
    x, y = [], []
    for a, b in best_edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=BEST_COLOR, width=2.5, dash="dot"), name="beste Tour bisher"))

    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=STOP_COLOR, line=dict(width=1, color="white")), name="Stopps"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.98))
    return _map_layout(fig)


def build_bounds_curve(generations):
    xs = list(range(1, len(generations) + 1))
    t_max = [g.t_max for g in generations]
    t_min = [g.t_min for g in generations]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=t_max, mode="lines", line=dict(color=MMAS_COLOR, width=2.5), name="τ_max"))
    fig.add_trace(go.Scatter(x=xs, y=t_min, mode="lines", line=dict(color=MMAS_COLOR, width=2.5, dash="dot"), name="τ_min"))
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Pheromon-Grenze", type="log")
    return _base(fig, 260)


def build_best_curve(best_history, reference=None):
    xs = list(range(1, len(best_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=best_history, mode="lines", line=dict(color=MMAS_COLOR, width=2.5), name="beste Tour bisher"))
    if reference is not None and np.isfinite(reference):
        fig.add_hline(y=reference, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Brute-Force-Optimum", annotation_position="bottom right")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Tourlänge")
    return _base(fig, 260)


def build_stagnation_comparison(report):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Locker<br>r={report['loose']['ratio']:.2f}"], y=[report["loose"]["near_max_share"]], marker_color=LOOSE_COLOR, showlegend=False, width=0.4))
    fig.add_trace(go.Bar(x=[f"Eng<br>r={report['tight']['ratio']:.2f}"], y=[report["tight"]["near_max_share"]], marker_color=TIGHT_COLOR, showlegend=False, width=0.4))
    fig.update_yaxes(title_text="Anteil Kanten nahe τ_max", range=[0, 1], tickformat=".0%")
    return _base(fig, 360)


def build_ratio_experiment(rows):
    xs = [r["ratio"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["length_median"] for r in rows], mode="lines+markers", line=dict(color=MMAS_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="τ-Verhältnis r (τ_min/τ_max)")
    fig.update_yaxes(title_text="Tourlänge (Median)")
    return _base(fig, 300)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=MMAS_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zum Brute-Force-Optimum (%)")
    return _base(fig, 300)
