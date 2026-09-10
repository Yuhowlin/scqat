"""Tests for CrosstalkCompensatedBenchmarkEstimator in scqat."""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.crosstalk_compensated_benchmark import CrosstalkCompensatedBenchmarkEstimator


def test_calibrate_mode_synthetic():
    amps = np.linspace(0.0, 0.08, 17)
    phases = np.linspace(-np.pi, np.pi, 21)

    opt_amp = 0.035
    opt_phase = 0.5

    A, P = np.meshgrid(amps, phases, indexing="ij")
    # Residual excitation P1 bowl centered at opt_amp, opt_phase
    p1_map = 0.01 + 0.45 * (1.0 - np.exp(-((A - opt_amp) ** 2 / 0.001 + (P - opt_phase) ** 2 / 0.8)))

    rng = np.random.default_rng(456)
    p1_data = p1_map + rng.normal(0, 0.005, (len(amps), len(phases)))

    ds = xr.Dataset(
        {"state": (("cancel_amp", "init_phase"), p1_data)},
        coords={
            "cancel_amp": amps,
            "init_phase": phases,
        },
    )

    est = CrosstalkCompensatedBenchmarkEstimator()
    res = est.extract_parameters(ds)

    assert res["success"] is True
    assert res["mode"] == "calibrate"
    assert np.isclose(res["optimal_cancel_amp"], opt_amp, atol=0.005)
    assert np.isclose(res["optimal_init_phase"], opt_phase, atol=0.15)
    assert "suggested_updates" in res

    figs = est.generate_figures(ds, res)
    assert "summary" in figs


def test_benchmark_mode_synthetic():
    repetitions = np.array([0, 2, 4, 8, 16, 32, 64])
    conditions = ["isolated", "simultaneous", "compensated"]

    # True error rate per gate
    eps = {
        "isolated": 0.005,
        "simultaneous": 0.035,
        "compensated": 0.008,
    }

    rng = np.random.default_rng(123)
    data = np.empty((len(conditions), len(repetitions)), dtype=float)
    for c_idx, cond in enumerate(conditions):
        e = eps[cond]
        p1 = 0.5 * (1.0 - (1.0 - 2.0 * e) ** repetitions)
        data[c_idx, :] = np.clip(p1 + rng.normal(0, 0.005, len(repetitions)), 0.0, 1.0)

    ds = xr.Dataset(
        {"state": (("condition", "repetitions"), data)},
        coords={
            "condition": conditions,
            "repetitions": repetitions,
        },
    )

    est = CrosstalkCompensatedBenchmarkEstimator()
    res = est.extract_parameters(ds)

    assert res["success"] is True
    assert res["mode"] == "benchmark"
    assert res["mean_error_isolated"] < res["mean_error_simultaneous"]
    assert res["mean_error_compensated"] < res["mean_error_simultaneous"]
    assert res["mitigation_fraction"] > 0.7

    figs = est.generate_figures(ds, res)
    assert "summary" in figs


def test_calibrate_mode_multi_stage_figures():
    import json

    amps1 = np.linspace(0.0, 0.08, 11)
    phases1 = np.linspace(-np.pi, np.pi, 15)
    p_vals1 = np.random.uniform(0.1, 0.5, (11, 15)).tolist()

    amps2 = np.linspace(0.02, 0.04, 11)
    phases2 = np.linspace(-0.5, 0.5, 15)
    p_vals2 = np.random.uniform(0.05, 0.2, (11, 15)).tolist()

    history = [
        {
            "stage": 1,
            "repetitions": 2,
            "cancel_amps": amps1.tolist(),
            "init_phases": phases1.tolist(),
            "p_vals": p_vals1,
            "best_cancel_amp": 0.03,
            "best_init_phase": 0.0,
            "min_signal": 0.1,
        },
        {
            "stage": 2,
            "repetitions": 8,
            "cancel_amps": amps2.tolist(),
            "init_phases": phases2.tolist(),
            "p_vals": p_vals2,
            "best_cancel_amp": 0.032,
            "best_init_phase": 0.05,
            "min_signal": 0.05,
        },
    ]

    ds = xr.Dataset(
        {"state": (("cancel_amp", "init_phase"), np.array(p_vals2))},
        coords={
            "cancel_amp": amps2,
            "init_phase": phases2,
        },
        attrs={"stage_history": json.dumps(history)},
    )

    est = CrosstalkCompensatedBenchmarkEstimator()
    res = est.extract_parameters(ds)
    figs = est.generate_figures(ds, res)

    assert "summary" in figs
    assert "stage_1" in figs
    assert "stage_2" in figs
