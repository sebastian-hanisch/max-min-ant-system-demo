"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session
(Permalink-Klammern und Preset-Knöpfe werden über AppTest in test_app.py geprüft, wie im Rest des Portfolios üblich)."""

import mmas_constants as C
import mmas_evaluation as E
import mmas_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX
        assert C.ANTS_MIN <= p["ants"] <= C.ANTS_MAX and C.GEN_MIN <= p["gens"] <= C.GEN_MAX
        assert C.RHO_MIN <= p["rho"] <= C.RHO_MAX
        assert C.RATIO_MIN <= p["ratio"] <= C.RATIO_MAX
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall"]
    s = E.Settings(n=p["n"], seed=p["seed"], ants=p["ants"], gens=p["gens"], alpha=p["alpha"], beta=p["beta"], rho=p["rho"], ratio=p["ratio"], run_seed=p["run_seed"])
    assert s == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX)
    assert P.bounds("rho_slider") == (C.RHO_MIN, C.RHO_MAX)
    assert P.bounds("ratio_slider") == (C.RATIO_MIN, C.RATIO_MAX)
    assert P.bounds("seed_input") == (0, C.SEED_MAX)
    assert set(P.STEPS) == {"n_slider", "ants_slider", "gens_slider", "alpha_slider", "beta_slider", "rho_slider", "ratio_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_comparison_preset_uses_the_shared_small_instance():
    p = C.PRESETS["Kleine Instanz (Vergleich mit Brute-Force)"]
    assert p["n"] == C.COMPARISON_N and p["seed"] == C.COMPARISON_VEHICLE_SEED


def test_ratio_presets_use_the_extreme_ratio_values():
    assert C.PRESETS["Enges τ-Verhältnis"]["ratio"] == C.RATIO_VALUES[0]
    assert C.PRESETS["Lockeres τ-Verhältnis"]["ratio"] == C.RATIO_VALUES[-1]
