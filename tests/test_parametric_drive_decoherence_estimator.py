"""Tests for the ParametricDriveDecoherenceEstimator.

The freq_time(_tomo) parametric-drive nodes measure rho_11(t) at several
``driving_frequency`` values. The estimator reconstructs rho_11 (full density
matrix for tomography data, rho_11-only otherwise), runs the per-frequency
Hankel -> multi-damped-osc -> non-Markovian decoherence pipeline, and reports
gamma / lambda / Delta and the EP figure of merit 8*lambda^2/gamma^2 vs frequency.

These tests synthesise clean rho_11(t) traces from the decoherence model and check
the output structure (frequency-resolved arrays + figures), both input layouts
(tomography via ``basis`` and rho_11-only), the metadata projection, and that at
least the decoherence stage converges on clean data.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import ParametricDriveDecoherenceEstimator
from scqat.estimators.parametric_drive_decoherence import (
    ParametricDriveDecoherenceEstimator as SubpkgEstimator,
)
from scqat.tools.fit_qubit_decoherence import rho11_model


def _rho11_traces(freqs, t, gammas, lam=2e-4, seed=0, noise=2e-3):
    """rho_11(t) for each driving frequency from the decoherence model."""
    rng = np.random.default_rng(seed)
    data = np.empty((len(freqs), t.size))
    for i, g in enumerate(gammas):
        clean = rho11_model(t, g, lam, 0.0, 1.0)
        data[i] = clean + noise * rng.standard_normal(t.size)
    return data


def _make_rho11_only(n_freq=3, n_time=80):
    """rho_11-only dataset (no basis) — mirrors the freq_time node."""
    freqs = np.linspace(330e6, 336e6, n_freq)
    t = np.linspace(0.0, 3000.0, n_time)
    gammas = np.linspace(1.5e-3, 3.0e-3, n_freq)
    data = _rho11_traces(freqs, t, gammas)
    ds = xr.Dataset(
        {"state": (("driving_frequency", "driving_time"), data)},
        coords={"driving_frequency": freqs, "driving_time": t},
    )
    return ds, gammas


def _make_tomo(n_freq=3, n_time=80):
    """Tomography dataset (basis = 0/1/2 for X/Y/Z) — mirrors the tomo node.

    Only the Z readout (basis=2) carries the rho_11 signal; X/Y are set to 0.5
    (rho_10 = 0), which is enough for the rho_11 decoherence fit.
    """
    ds11, gammas = _make_rho11_only(n_freq, n_time)
    rho11 = ds11["state"].values
    state = np.empty((n_freq, n_time, 3))
    state[:, :, 0] = 0.5  # X readout -> <sx> = 0
    state[:, :, 1] = 0.5  # Y readout -> <sy> = 0
    state[:, :, 2] = rho11  # Z readout -> rho_11 (offset 0, scale 1)
    ds = xr.Dataset(
        {"state": (("driving_frequency", "driving_time", "basis"), state)},
        coords={
            "driving_frequency": ds11["driving_frequency"].values,
            "driving_time": ds11["driving_time"].values,
            "basis": [0, 1, 2],
        },
    )
    return ds, gammas


# Offset 0 / scale 1 so the planted rho_11 is recovered verbatim.
_KW = dict(rho11_offset=0.0, rho11_scale=1.0)


class TestParametricDriveDecoherenceEstimator:

    def test_imports_match(self):
        assert ParametricDriveDecoherenceEstimator is SubpkgEstimator
        assert (
            ParametricDriveDecoherenceEstimator.estimator_name
            == "parametric_drive_decoherence"
        )

    def test_check_data_requires_coords_and_state(self):
        est = ParametricDriveDecoherenceEstimator()
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"state": ("driving_time", [0.0, 1.0])},
                                       coords={"driving_time": [0.0, 1.0]}))
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset(
                {"other": (("driving_frequency", "driving_time"), np.zeros((2, 2)))},
                coords={"driving_frequency": [0.0, 1.0], "driving_time": [0.0, 1.0]},
            ))

    def test_rho11_only_shapes_and_fit(self):
        ds, gammas = _make_rho11_only()
        res = ParametricDriveDecoherenceEstimator().extract_parameters(ds, **_KW)
        n = len(gammas)
        assert res["has_tomography"] is False
        assert res["n_freq"] == n
        for key in ("gamma", "lambda_", "Delta", "ep_metric"):
            assert np.asarray(res[key]).shape == (n,)
        assert res["rho11_data"].shape == (n, ds.sizes["driving_time"])
        # Clean data: the decoherence stage should converge at every frequency.
        assert res["n_decoh_ok"] == n
        assert np.isfinite(res["gamma"]).all()

    def test_default_offset_scale_is_identity(self):
        """With no offset/scale kwargs the estimator must not rescale a population
        dataset: rho_11 equals the input state verbatim (offset 0 / scale 1
        default). Guards against re-introducing the I-quadrature readout
        correction that pushed discriminated rho_11 outside [0, 1]."""
        ds, _ = _make_rho11_only()
        res = ParametricDriveDecoherenceEstimator().extract_parameters(ds)
        np.testing.assert_allclose(res["rho11_data"], ds["state"].values)

    def test_tomography_path(self):
        ds, gammas = _make_tomo()
        res = ParametricDriveDecoherenceEstimator().extract_parameters(ds, **_KW)
        assert res["has_tomography"] is True
        assert res["n_freq"] == len(gammas)
        assert res["n_decoh_ok"] >= 1

    def test_single_frequency_rho11_only(self):
        """frequency_points=1 keeps 'driving_frequency' as a length-1 dim instead of
        squeezing it away (regression for the rho_11-only analysis crash)."""
        ds, _ = _make_rho11_only(n_freq=1)
        res = ParametricDriveDecoherenceEstimator().extract_parameters(ds, **_KW)
        assert res["n_freq"] == 1
        assert np.asarray(res["gamma"]).shape == (1,)

    def test_single_frequency_tomography(self):
        """Same single-frequency regression for the tomography (basis) path."""
        ds, _ = _make_tomo(n_freq=1)
        res = ParametricDriveDecoherenceEstimator().extract_parameters(ds, **_KW)
        assert res["n_freq"] == 1

    def test_metadata_drops_bulky(self):
        ds, _ = _make_rho11_only()
        est = ParametricDriveDecoherenceEstimator()
        res = est.extract_parameters(ds, **_KW)
        meta = est.extract_metadata(res)
        for k in ("rho11_data", "rho11_fit", "hankel", "mdo", "decoh", "decoh_guesses",
                  "driving_time"):
            assert k not in meta
        assert {"driving_frequency", "gamma", "lambda_", "ep_metric", "success"} <= set(meta)

    def test_plot_data_layout(self):
        ds, _ = _make_rho11_only()
        est = ParametricDriveDecoherenceEstimator()
        res = est.extract_parameters(ds, **_KW)
        pd = est.build_plot_data(ds, res)
        assert pd["rho11_data"].dims == ("driving_frequency", "driving_time")
        assert pd["gamma"].dims == ("driving_frequency",)
        assert "ep_metric" in pd

    def test_figures_render_on_a_failed_fit(self, tmp_path):
        """Pure noise makes the three-stage pipeline degenerate, but the raw
        rho_11 traces must still be drawn: both figures render and both PNGs
        land. `build_plot_data` degrades a failed fit to NaN rather than
        dropping the field, and `plot_rho11_fits` guards its overlay."""
        freqs = np.linspace(330e6, 336e6, 3)
        t = np.linspace(0.0, 3000.0, 60)
        rng = np.random.default_rng(7)
        noise = rng.normal(0.5, 0.05, (freqs.size, t.size))
        ds = xr.Dataset(
            {"state": (("driving_frequency", "driving_time"), noise)},
            coords={"driving_frequency": freqs, "driving_time": t},
        )
        est = ParametricDriveDecoherenceEstimator()
        _, figs = est.analyze(ds, output_dir=str(tmp_path), **_KW)
        assert set(figs) == {"decoherence_params", "rho11_fits"}
        assert (tmp_path / "parametric_drive_decoherence_rho11_fits.png").exists()
        plt.close("all")

    def test_a_broken_plotter_does_not_drop_its_sibling(self, monkeypatch):
        """The isolation itself: the pure-FIT panel raising must not take the
        raw-carrying rho11_fits figure down with it. SCQO's artifact fallback
        drops ALL figures on any single plotter exception, so one broken panel
        would otherwise cost the run every PNG."""
        import scqat.estimators.parametric_drive_decoherence.estimator as mod

        def boom(_plot_data):
            raise RuntimeError("degenerate fit panel")

        monkeypatch.setattr(mod, "plot_decoherence_params", boom)
        ds, _ = _make_rho11_only(n_freq=2, n_time=40)
        est = ParametricDriveDecoherenceEstimator()
        res = est.extract_parameters(ds, **_KW)
        with pytest.warns(UserWarning, match="decoherence_params"):
            figs = est.generate_figures(ds, res)
        assert set(figs) == {"rho11_fits"}
        plt.close("all")

    def test_analyze_roundtrip(self, tmp_path):
        ds, _ = _make_rho11_only()
        est = ParametricDriveDecoherenceEstimator()
        res, figs = est.analyze(ds, output_dir=str(tmp_path), **_KW)
        assert (tmp_path / "parametric_drive_decoherence_metadata.json").exists()
        assert (tmp_path / "parametric_drive_decoherence_plotdata.nc").exists()
        assert set(figs) == {"decoherence_params", "rho11_fits"}
        assert isinstance(figs["decoherence_params"], plt.Figure)
        assert isinstance(figs["rho11_fits"], plt.Figure)
        plt.close("all")
