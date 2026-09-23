"""Max-Min Ant System - begrenztes Pheromon statt unbegrenzter Verstärkung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, FIX-Nachfolger von ant-system-demo:
Max-Min Ant System (Stützle & Hoos, 1996/2000) behebt Ant Systems genannte Schwäche - keine Pheromon-Obergrenze,
Stagnationsgefahr - über zwei Änderungen: Pheromonwerte werden auf [τ_min, τ_max] begrenzt, und nur die BESTE
Ameise einer Generation legt Pheromon ab. Vehikel ist dieselbe diskrete Lieferroute wie ant-system-demo.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import mmas_constants as C
from mmas_evaluation import Settings, analyse, ratio_experiment, stagnation_experiment, sweep
from mmas_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from mmas_visualization import build_best_curve, build_bounds_curve, build_pheromone_map, build_ratio_experiment, build_sweep, build_stagnation_comparison

st.set_page_config(page_title="Max-Min Ant System – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _stagnation():
    return stagnation_experiment()


@st.cache_data(show_spinner=False)
def _ratio_experiment():
    return ratio_experiment()


st.title("🐜 Max-Min Ant System – begrenztes Pheromon statt unbegrenzter Verstärkung")
st.markdown(
    """
Bei Ant System kann ein früh gefundener, mittelmäßiger Pfad sich unbegrenzt verstärken - die ganze Kolonie legt
sich vorzeitig darauf fest (Stagnation). **Max-Min Ant System** (Stützle & Hoos, 1996/2000) behebt genau das über
zwei Änderungen: Pheromonwerte werden auf ein Band **[τ_min, τ_max]** begrenzt, und nur die **beste Ameise** einer
Generation legt Pheromon ab (statt alle) - beides zusammen verhindert, dass ein einzelner Pfad die ganze Kolonie
dauerhaft dominiert.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "zehntes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, ein **Fix** für "
    "[ant-system-demo](https://github.com/sebastian-hanisch/ant-system-demo) - "
    "**ein** Verfahren an einem wachsenden Beispiel. Vehikel ist dieselbe diskrete Lieferroute wie dort/bei "
    "[nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/)."
)

with st.expander("So funktioniert Max-Min Ant System", expanded=True):
    st.markdown(
        r"""
1. **Tourkonstruktion.** Wie bei Ant System: jede Ameise baut eine Tour, Übergangswahrscheinlichkeit
   $p_{ij} \propto \tau_{ij}^\alpha \eta_{ij}^\beta$.
2. **Nur die beste Ameise legt ab.** Nach jeder Generation aktualisiert NUR die beste Ameise DIESER Generation die
   Pheromonspur - nicht alle wie bei Ant System.
3. **Begrenztes Band.** Nach jedem Update wird Pheromon auf $[\tau_{min}, \tau_{max}]$ geklemmt -
   $\tau_{max} = 1/(\rho \cdot L_{best})$, $\tau_{min} = \tau_{max} \cdot r$ (τ-Verhältnis $r$, der eigene Regler
   dieser Demo).
4. **Kein Pfad kann sich unbegrenzt verstärken.** Selbst der bisher beste Pfad kann höchstens $\tau_{max}$
   erreichen - und jeder andere Pfad bleibt mindestens bei $\tau_{min}$, bleibt also potenziell wählbar.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
    st.markdown("**Max-Min Ant System**")
    ants = st.slider("Ameisenzahl", *bounds("ants_slider"), key="ants_slider", step=C.ANTS_STEP)
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    alpha = st.slider("Pheromon-Exponent α", *bounds("alpha_slider"), key="alpha_slider", step=C.ALPHA_STEP)
    beta = st.slider("Sichtbarkeits-Exponent β", *bounds("beta_slider"), key="beta_slider", step=C.BETA_STEP)
    rho = st.slider("Verdunstungsrate ρ", *bounds("rho_slider"), key="rho_slider", step=C.RHO_STEP)
    ratio = st.slider("τ-Verhältnis r (τ_min/τ_max)", *bounds("ratio_slider"), key="ratio_slider", step=C.RATIO_STEP, format="%.3f", help="Wie eng das erlaubte Pheromon-Band ist - klein = locker (nahe Ant System), groß = eng.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des MMAS-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "n_slider": int(n_stops), "ants_slider": int(ants), "gens_slider": int(generations),
    "alpha_slider": float(alpha), "beta_slider": float(beta), "rho_slider": float(rho), "ratio_slider": float(ratio),
    "seed_input": int(seed), "run_seed_input": int(run_seed),
})

settings = Settings(n=int(n_stops), seed=int(seed), ants=int(ants), gens=int(generations), alpha=float(alpha), beta=float(beta), rho=float(rho), ratio=float(ratio), run_seed=int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations)
data_key = settings

# --- MMAS in Aktion ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 MMAS in Aktion")
if "mmas_gen" not in st.session_state or st.session_state.get("mmas_gen_owner") != data_key:
    st.session_state["mmas_gen"] = n_gens_run
    st.session_state["mmas_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 1, n_gens_run, key="mmas_gen")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    return sorted({int(round(x)) for x in np.linspace(1, n_gens_run, min(n_gens_run, 40))})


def _render(g):
    gd = result.generations[g - 1]
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Generation {g} von {n_gens_run} – beste Tour bisher: {result.best_history[g - 1]:.1f} km – τ∈[{gd.t_min:.4f}, {gd.t_max:.4f}]**")
        c1.plotly_chart(build_pheromone_map(a.inst.xy, gd), width="stretch", key=f"g_map_{g}")
        c2.markdown("**Beste Tour bisher**")
        c2.plotly_chart(build_best_curve(result.best_history[:g], reference=a.brute_force_optimum), width="stretch", key=f"g_best_{g}")


if auto_play:
    for fr in _frames():
        _render(fr)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was MMAS gefunden hat")
m1, m2 = st.columns(2)
m1.metric("Beste gefundene Tour", f"{result.best_length:.1f} km")
if np.isfinite(a.brute_force_optimum):
    m2.metric("Abstand zum Brute-Force-Optimum", f"{a.gap:+.1f} %")
else:
    m2.metric("Brute-Force-Optimum", "nicht berechenbar (zu viele Stopps)")
st.markdown("**τ_min/τ_max über die Generationen**")
st.plotly_chart(build_bounds_curve(result.generations), width="stretch")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Abstand zum Optimum von Ameisenzahl und τ-Verhältnis ab?")
st.caption("Läuft auf der kleinen Vergleichsinstanz (8 Stopps) mit eigenem knappen Budget - beim komfortablen Kopfexperiment-Budget trifft fast jede Einstellung das Optimum.")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, None)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: Kopfexperiment - validiert den Fix (selbst gemessen) ------------------------------------------------------------------------

st.subheader("🔬 Verhindert die Begrenzung wirklich Stagnation?")
st.caption(f"Ein langer Lauf ({C.STAGNATION_GENS} Generationen) - Anteil der Pheromonwerte, die nahe τ_max „kleben“ (ein Stagnations-Symptom), bei lockerem gegen eng gewähltem τ-Verhältnis r.")
if st.button("Locker gegen eng gewähltes τ-Verhältnis rechnen (dauert etwa 5 Sekunden)", key="stagnation_start"):
    st.session_state["stagnation_on"] = True
if st.session_state.get("stagnation_on"):
    with st.spinner("Rechne zwei lange MMAS-Läufe..."):
        report = _stagnation()
    st.plotly_chart(build_stagnation_comparison(report), width="stretch", key="stagnation_chart")
    st.warning(
        "**Ehrlicher Befund:** Bei lockerem τ-Verhältnis (τ_min nahe 0, näher am unbeschränkten Ant-System-Verhalten) "
        "'klebt' ein klar messbarer Anteil der Pheromonwerte nahe τ_max - genau das Stagnations-Symptom, das Ant "
        "System fehlte zu verhindern. Bei eng gewähltem τ-Verhältnis verschwindet dieser Effekt praktisch vollständig. "
        "Das validiert den Fix direkt an MMAS' eigener Kernbehauptung."
    )

st.markdown("---")

# --- Experiment 2: eigener Regler - tau-Verhältnis r --------------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hängt die Tourqualität vom τ-Verhältnis r ab?")
st.caption("Zu eng: Pheromon kann kaum differenzieren, die Suche wird fast gleichverteilt. Zu locker: das Stagnationsrisiko kehrt zurück (siehe Experiment oben).")
if st.button(f"τ-Verhältnisse {C.RATIO_VALUES[0]:.3f} bis {C.RATIO_VALUES[-1]:.2f} vergleichen (dauert etwa 10 Sekunden)", key="ratio_start"):
    st.session_state["ratio_on"] = True
if st.session_state.get("ratio_on"):
    with st.spinner("Rechne 5 τ-Verhältnisse × 20 Läufe..."):
        rows_ratio = _ratio_experiment()
    st.plotly_chart(build_ratio_experiment(rows_ratio), width="stretch", key="ratio_chart")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Vereinfachtes τ_min = τ_max · r** | Stützle & Hoos' Originalarbeit leitet τ_min analytisch über die Konvergenzwahrscheinlichkeit p_best her - diese Demo nutzt bewusst ein festes Verhältnis statt der vollen Herleitung. | Hier nicht umgesetzt, siehe README |
| **Keine Stagnations-Erkennung mit Reinitialisierung** | Die vollständige MMAS-Variante erkennt Stagnation über den λ-Verzweigungsfaktor und setzt Pheromon dann aktiv auf τ_max zurück - hier nicht umgesetzt, die Grenzen allein müssen reichen. | Hier nicht umgesetzt |
| **Nur iteration-best, keine Ablage-Alternation** | Fortgeschrittenere MMAS-Varianten wechseln zwischen iteration-best und global-best-Ablage - diese Demo nutzt durchgehend iteration-best. | Hier nicht umgesetzt |
| **τ-Verhältnis r ist gut gewählt** | Zu eng: Tourqualität leidet messbar (siehe Experiment oben). Zu locker: Stagnationsrisiko kehrt zurück. | Muss von Hand eingestellt werden, wie bei jedem Regler dieser Linie |
"""
)
st.caption(
    "Max-Min Ant System schließt die Ant-System-Kette dieser Linie ab (kein Nachfolger geplant). Vorgänger: "
    "[ant-system-demo](https://github.com/sebastian-hanisch/ant-system-demo), dessen dokumentierte Stagnationsgefahr hier direkt geprüft wird."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Übergangswahrscheinlichkeit** (wie Ant System). $p_{ij} = \dfrac{\tau_{ij}^\alpha \eta_{ij}^\beta}{\sum_{k \in
\text{erlaubt}} \tau_{ik}^\alpha \eta_{ik}^\beta}$.

**Nur-beste-Ameise-Update.** $\tau_{ij} \leftarrow (1-\rho)\tau_{ij} + \Delta\tau_{ij}^{best}$,
$\Delta\tau_{ij}^{best} = Q/L_{best}$ nur für Kanten der besten Tour DIESER Generation.

**Grenzen.** $\tau_{ij} \leftarrow \text{clip}(\tau_{ij}, \tau_{min}, \tau_{max})$, mit
$\tau_{max} = 1/(\rho \cdot L_{best,global})$, $\tau_{min} = \tau_{max} \cdot r$.

Implementiert in `mmas_algorithm.py` (Übergangswahrscheinlichkeit, Tourkonstruktion, Best-Ant-Update, Grenzen,
Hauptschleife), `mmas_scenario.py` (Vehikel), `mmas_evaluation.py` (Kennzahlen, Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
