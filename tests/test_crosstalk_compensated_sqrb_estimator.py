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

