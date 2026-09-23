"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Generation-Slider inkl. Abspielen ohne doppelte Schlüssel,
Würfel-Knöpfe, Permalink-Grenzen, beide Experimente + Sweep auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mmas_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_has_no_exception_and_shows_metrics():
    at = _run()
    _ok(at)
    assert at.metric


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["ratio_slider"] == p["ratio"]
    assert at.metric


def test_generation_slider_runs_at_various_positions():
    at = _run(gens_slider=20, n_slider=10, ants_slider=6)
    _ok(at)
    gen_slider = next(s for s in at.slider if s.key == "mmas_gen")
    gen_slider.set_value(1).run()
    _ok(at)
    assert at.get("plotly_chart")
    gen_slider.set_value(20).run()
    _ok(at)
    assert at.get("plotly_chart")


def test_play_runs_without_duplicate_keys():
    at = _run(gens_slider=20, n_slider=10, ants_slider=6)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_dice_buttons_change_the_seeds():
    at = _run()
    old_seed = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Vehikel generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old_seed
    old_run_seed = at.session_state["run_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["run_seed_input"] != old_run_seed


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["rho"] = "9999"
    at.query_params["n"] = "13"
    at.run()
    _ok(at)
    assert at.session_state["rho_slider"] == C.RHO_MAX
    assert at.session_state["n_slider"] == 12


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(ants_slider=C.ANTS_MIN), dict(gens_slider=C.GEN_MIN), dict(alpha_slider=C.ALPHA_MIN), dict(beta_slider=C.BETA_MAX), dict(rho_slider=C.RHO_MIN), dict(rho_slider=C.RHO_MAX), dict(ratio_slider=C.RATIO_MIN), dict(ratio_slider=C.RATIO_MAX)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_sweep_runs_on_demand():
    at = _run()
    at.selectbox(key="sweep_select").set_value("ratio").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_stagnation_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "STAGNATION_GENS", 15)
    at = _run()
    next(b for b in at.button if b.key == "stagnation_start").click().run()
    _ok(at)
    assert at.session_state["stagnation_on"] and at.get("plotly_chart")


def test_ratio_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "RATIO_VALUES", (0.02, 0.4))
    monkeypatch.setattr(C, "RATIO_EXPERIMENT_SEEDS", (1, 2))
    monkeypatch.setattr(C, "RATIO_EXPERIMENT_ANTS", 6)
    monkeypatch.setattr(C, "RATIO_EXPERIMENT_GENS", 10)
    at = _run()
    next(b for b in at.button if b.key == "ratio_start").click().run()
    _ok(at)
    assert at.session_state["ratio_on"] and at.get("plotly_chart")


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("unbegrenzt verstärken" in m.value for m in at.markdown)
