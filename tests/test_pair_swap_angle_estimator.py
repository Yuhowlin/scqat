"""Synthetic tests for the partial-swap angle calibration estimator.

The dataset is one pair's joint populations over (coupler flux, swap count). The
planted physics is a coherent partial exchange: applying an angle ``theta`` N
times transfers ``sin^2(N*theta)``, so the transfer oscillates in N with period
``pi/theta`` and the fit must recover ``theta`` per coupler-flux row.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_swap_angle import PairSwapAngleEstimator
from scqat.estimators.pair_swap_angle.estimator import FIG_ANGLE, FIG_MAP

#: pair basis labels, digit order (high, low).
LABELS = ["00", "01", "10", "11"]

#: driving the LOW member prepares "01"; the excitation transfers to "10".
KWARGS = {"drive_side": "low", "flux_side": "low",
          "high_name": "q2", "low_name": "q1"}


def _angle_ds(thetas=None, n_max: int = 24, noise: float = 0.0) -> xr.Dataset:
    """A coherent partial swap whose angle grows with the coupler flux.

    ``theta`` is planted linearly against the knob only so the test has a known
    answer; the estimator makes no monotonicity assumption.
    """
    thetas = np.linspace(0.3, 1.2, 9) if thetas is None else np.asarray(thetas)
    knob = np.linspace(0.05, 0.45, thetas.size)
    counts = np.arange(0, n_max + 1)

    transfer = np.sin(np.outer(thetas, counts)) ** 2          # (knob, N)
    if noise:
        rng = np.random.default_rng(0)
        transfer = np.clip(transfer + rng.normal(0.0, noise, transfer.shape), 0.0, 1.0)
    prepared = 1.0 - transfer

    rows = {
        "00": np.zeros_like(transfer),
        "01": prepared,      # low excited - the prepared state
        "10": transfer,      # high excited - the transfer state
        "11": np.zeros_like(transfer),
    }
    jp = np.stack([rows[label] for label in LABELS])          # (joint, knob, N)
    return xr.Dataset(
        {"joint_population": (("joint_state", "coupler_flux_v", "swap_count"), jp)},
        coords={"joint_state": LABELS, "coupler_flux_v": knob, "swap_count": counts},
    )


def test_recovers_the_planted_angle_per_knob_value():
    est = PairSwapAngleEstimator()
    thetas = np.linspace(0.3, 1.2, 9)
    ds = _angle_ds(thetas)
    est._check_data(ds)
    res = est.extract_parameters(ds, **KWARGS)

    assert res["n_theta_ok"] == thetas.size
    assert res["theta_success"] == [1] * thetas.size
    np.testing.assert_allclose(res["theta_rad"], thetas, atol=5e-3)
    # period = pi/theta is the relation TUTORIAL section 12 states
    np.testing.assert_allclose(
        res["swap_period"], np.pi / thetas, rtol=5e-3
    )


def test_reports_the_reachable_angle_range():
    est = PairSwapAngleEstimator()
    ds = _angle_ds()
    res = est.extract_parameters(ds, **KWARGS)
    assert res["theta_min_rad"] == pytest.approx(0.3, abs=5e-3)
    assert res["theta_max_rad"] == pytest.approx(1.2, abs=5e-3)


def test_solves_the_curve_for_a_requested_angle():
    """A target between two measured rows is interpolated, not snapped."""
    est = PairSwapAngleEstimator()
    thetas = np.linspace(0.3, 1.2, 9)
    ds = _angle_ds(thetas)
    knob = np.asarray(ds["coupler_flux_v"].values)
    # halfway between two grid angles, so a nearest-row answer would be wrong
    target = 0.5 * (thetas[3] + thetas[4])
    res = est.extract_parameters(ds, target_theta_rad=target, **KWARGS)

    assert res["best_is_interpolated"] == 1
    assert res["best_theta_rad"] == pytest.approx(target)
    assert res["best_coupler_flux_v"] == pytest.approx(
        0.5 * (knob[3] + knob[4]), abs=2e-3
    )


def test_unreachable_target_degrades_to_the_nearest_measured_row():
    est = PairSwapAngleEstimator()
    ds = _angle_ds()
    knob = np.asarray(ds["coupler_flux_v"].values)
    res = est.extract_parameters(ds, target_theta_rad=3.0, **KWARGS)

    assert res["best_is_interpolated"] == 0
    # the largest angle in the sweep is the closest approach to pi
    assert res["best_coupler_flux_v"] == pytest.approx(knob[-1])
    assert res["best_theta_rad"] == pytest.approx(1.2, abs=5e-3)


def test_no_target_leaves_the_solution_nan():
    est = PairSwapAngleEstimator()
    res = PairSwapAngleEstimator().extract_parameters(_angle_ds(), **KWARGS)
    assert np.isnan(res["target_theta_rad"])
    assert np.isnan(res["best_coupler_flux_v"])
    assert est.estimator_name == "pair_swap_angle"


def test_flat_row_is_rejected_not_reported_as_an_angle():
    """A knob value with no exchange must fail its fit, not invent a theta."""
    est = PairSwapAngleEstimator()
    ds = _angle_ds()
    # kill the exchange at the first knob value: all population stays prepared
    ds["joint_population"][LABELS.index("10"), 0, :] = 0.0
    ds["joint_population"][LABELS.index("01"), 0, :] = 1.0
    res = est.extract_parameters(ds, **KWARGS)

    assert res["theta_success"][0] == 0
    assert np.isnan(res["theta_rad"][0])
    # the rest of the curve is unaffected
    assert all(res["theta_success"][1:])


def test_metadata_drops_the_bulky_maps_but_keeps_the_curve():
    est = PairSwapAngleEstimator()
    ds = _angle_ds()
    res = est.extract_parameters(ds, **KWARGS)
    meta = est.extract_metadata(res)

    assert "_transfer" not in meta and "_best_fit" not in meta
    assert len(meta["theta_rad"]) == ds.sizes["coupler_flux_v"]
    assert meta["drive_side"] == "low"


def test_plot_data_carries_the_raw_transfer_map():
    est = PairSwapAngleEstimator()
    ds = _angle_ds()
    res = est.extract_parameters(ds, **KWARGS)
    pd = est.build_plot_data(ds, res, **KWARGS)

    assert pd["transfer"].dims == ("coupler_flux_v", "swap_count")
    assert pd["theta_rad"].dims == ("coupler_flux_v",)
    # netCDF-safe: no bools, no complex
    for name in pd.data_vars:
        assert pd[name].dtype.kind in "fiu", name
    assert pd.attrs["n_theta_ok"] == res["n_theta_ok"]


def test_figures_render_on_a_failed_acquisition():
    """An all-NaN map must still produce BOTH figures, not zero of them."""
    est = PairSwapAngleEstimator()
    ds = _angle_ds()
    ds["joint_population"] = ds["joint_population"] * np.nan

    res = est.extract_parameters(ds, **KWARGS)
    assert res["n_theta_ok"] == 0
    assert np.isnan(res["best_coupler_flux_v"])
    plot_data = est.build_plot_data(ds, res, **KWARGS)
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {
        FIG_MAP,
        FIG_ANGLE,
    }


def test_figures_render_from_plot_data_alone():
    """The replot path: no results dict, only a saved plot_data."""
    est = PairSwapAngleEstimator()
    ds = _angle_ds()
    res = est.extract_parameters(ds, target_theta_rad=0.7, **KWARGS)
    plot_data = est.build_plot_data(ds, res, **KWARGS)
    assert set(est.generate_figures(None, {}, plot_data=plot_data)) == {
        FIG_MAP,
        FIG_ANGLE,
    }


def test_noise_does_not_break_the_angle_fit():
    est = PairSwapAngleEstimator()
    thetas = np.linspace(0.3, 1.2, 9)
    res = est.extract_parameters(_angle_ds(thetas, noise=0.02), **KWARGS)
    np.testing.assert_allclose(res["theta_rad"], thetas, atol=0.02)
