"""Tests for CrosstalkCompensatedSQRBEstimator."""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBEstimator


def test_benchmark_mode_synthetic():
    depths = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    conditions = ["isolated", "simultaneous", "compensated"]
    n_seq = 20

    # True survival factors
    p_true = {
        "isolated": 0.994,
        "simultaneous": 0.965,
        "compensated": 0.990,
    }

    rng = np.random.default_rng(123)
    data = np.empty((len(conditions), n_seq, len(depths)), dtype=float)
    for c_idx, cond in enumerate(conditions):
        p_val = p_true[cond]
        for d_idx, d in enumerate(depths):
            mean_p = 0.5 + 0.5 * (p_val ** d)
            data[c_idx, :, d_idx] = mean_p + rng.normal(0, 0.005, n_seq)

    ds = xr.Dataset(
        {"state": (("condition", "sequence_idx", "depth"), data)},
        coords={
            "condition": conditions,
            "sequence_idx": np.arange(n_seq),
            "depth": depths,
        },
    )

    est = CrosstalkCompensatedSQRBEstimator()
    res = est.extract_parameters(ds, mode="benchmark")

    assert res["success"] is True
    assert res["r_isolated"] < res["r_simultaneous"]
    assert res["r_compensated"] < res["r_simultaneous"]
    assert res["mitigation_ratio"] > 0.6  # >60% error recovery

    # Check figures and plot data
    plot_data = est.build_plot_data(ds, res, mode="benchmark")
    figs = est.generate_figures(ds, res, plot_data, mode="benchmark")
    assert "crosstalk_compensated_sqrb" in figs


def test_calibrate_mode_synthetic():
    amps = np.linspace(0.0, 0.08, 17)
    phases = np.linspace(-np.pi, np.pi, 21)
    n_seq = 10

    opt_amp = 0.035
    opt_phase = 0.5

    A, P = np.meshgrid(amps, phases, indexing="ij")
    p0_map = 0.5 + 0.45 * np.exp(-((A - opt_amp) ** 2 / 0.001 + (P - opt_phase) ** 2 / 0.8))

    rng = np.random.default_rng(456)
    p0_data = np.tile(p0_map[:, :, None], (1, 1, n_seq)) + rng.normal(0, 0.005, (len(amps), len(phases), n_seq))
    # State variable on hardware represents P1 = 1 - P0
    p1_data = 1.0 - p0_data

    ds = xr.Dataset(
        {"state": (("cancel_amp", "init_phase", "sequence_idx"), p1_data)},
        coords={
            "cancel_amp": amps,
            "init_phase": phases,
            "sequence_idx": np.arange(n_seq),
        },
    )

    est = CrosstalkCompensatedSQRBEstimator()
    res = est.extract_parameters(ds, mode="calibrate")

    assert res["success"] is True
    assert np.isclose(res["optimal_cancel_amp"], opt_amp, atol=0.005)
    assert np.isclose(res["optimal_init_phase_rad"], opt_phase, atol=0.15)

    plot_data = est.build_plot_data(ds, res, mode="calibrate")
    figs = est.generate_figures(ds, res, plot_data, mode="calibrate")
    assert "crosstalk_compensation_calibration" in figs


def test_population_variable_support():
    # Test that estimator accepts population variable in both calibrate and benchmark modes
    depths = np.array([1, 2, 4, 8, 16])
    conditions = ["isolated", "simultaneous", "compensated"]
    n_seq = 5
    data = np.ones((len(conditions), n_seq, len(depths)), dtype=float) * 0.95

    ds_bench = xr.Dataset(
        {"population": (("condition", "sequence_idx", "depth"), data)},
        coords={
            "condition": conditions,
            "sequence_idx": np.arange(n_seq),
            "depth": depths,
        },
    )
    est = CrosstalkCompensatedSQRBEstimator()
    res = est.extract_parameters(ds_bench, mode="benchmark")
    assert res["success"] is True

    ds_cal = xr.Dataset(
        {"population": (("cancel_amp", "init_phase", "sequence_idx"), np.ones((5, 5, 2)) * 0.1)},
        coords={
            "cancel_amp": np.linspace(0, 0.05, 5),
            "init_phase": np.linspace(-np.pi, np.pi, 5),
            "sequence_idx": np.arange(2),
        },
    )
    res_cal = est.extract_parameters(ds_cal, mode="calibrate")
    assert res_cal["success"] is True


def test_calibrate_mode_raw_i():
    amps = np.linspace(0.0, 0.08, 17)
    phases = np.linspace(-np.pi, np.pi, 21)
    n_seq = 4

    opt_amp = 0.04
    opt_phase = -1.2

    A, P = np.meshgrid(amps, phases, indexing="ij")
    # Raw voltage in millivolts with baseline 0.010 V, optimum at 0.025 V
    i_map = 0.010 + 0.015 * np.exp(-((A - opt_amp) ** 2 / 0.001 + (P - opt_phase) ** 2 / 0.8))

    ds = xr.Dataset(
        {"I": (("cancel_amp", "init_phase", "sequence_idx"), np.tile(i_map[:, :, None], (1, 1, n_seq)))},
        coords={
            "cancel_amp": amps,
            "init_phase": phases,
            "sequence_idx": np.arange(n_seq),
        },
    )
    est = CrosstalkCompensatedSQRBEstimator()
    res = est.extract_parameters(ds, mode="calibrate")
    assert res["success"] is True
    assert np.isclose(res["optimal_cancel_amp"], opt_amp, atol=0.005)
    assert np.isclose(res["optimal_init_phase_rad"], opt_phase, atol=0.15)


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
            "repetitions": 5,
            "cancel_amps": amps1.tolist(),
            "init_phases": phases1.tolist(),
            "p_vals": p_vals1,
            "best_cancel_amp": 0.03,
            "best_init_phase": 0.0,
            "min_signal": 0.1,
        },
        {
            "stage": 2,
            "repetitions": 20,
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

    est = CrosstalkCompensatedSQRBEstimator()
    res = est.extract_parameters(ds)

    assert res["success"] is True
    assert "suggested_updates" in res
    assert "cancel_amp" in res["suggested_updates"]
    assert "init_phase" in res["suggested_updates"]

    figs = est.generate_figures(ds, res)
    assert "summary" in figs
    assert "stage_1" in figs
    assert "stage_2" in figs



