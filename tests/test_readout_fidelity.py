"""Tests for the readout-fidelity sweep estimators.

Synthesises a swept state-discrimination experiment — two IQ blobs (prepared states
0 and 1) whose separation peaks at a known sweep value — and checks that each
dedicated estimator reports that point as ``best_sweep_value``, exposes the
``best_*`` / ``success`` fields the LCHQMDriver nodes rely on, drops the bulky
per-slice arrays from its metadata, and (for power) honours ``outliers_threshold``.

Construction note: ``StateDiscriminationEstimator`` fixes each GMM centre at the
per-state histogram peak, so label index == prepared_state index and the fidelity
(mean of the ``direct_counts`` diagonal) rises with blob separation. A Gaussian bump
in the separation profile therefore makes the fidelity unimodal in the sweep.
"""

import json

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import (
    ReadoutFidelityEstimator,
    ReadoutFreqFidelityEstimator,
    ReadoutPowerFidelityEstimator,
)
from scqat.estimators.readout_fidelity import (
    ReadoutFreqFidelityEstimator as SubpkgFreq,
    ReadoutPowerFidelityEstimator as SubpkgPower,
)
from scqat.core.base_estimator import _json_safe


def _make_sweep_ds(coord, n_sweep=7, best_idx=4, n_shots=600, sigma=1.0, seed=0):
    """Swept two-blob discrimination data whose separation (and hence fidelity)
    peaks at ``best_idx``. ``coord`` names the swept axis (e.g. ``frequency`` /
    ``amp_prefactor``); its values are arbitrary but monotonic."""
    rng = np.random.default_rng(seed)
    s = np.arange(n_sweep)
    sep = 1.0 + 2.5 * np.exp(-0.5 * ((s - best_idx) / 1.2) ** 2)  # bump, peak at best_idx

    I = np.empty((n_sweep, 2, n_shots))
    Q = np.empty((n_sweep, 2, n_shots))
    for i in range(n_sweep):
        for state, centre_I in enumerate((0.0, sep[i])):
            I[i, state] = rng.normal(centre_I, sigma, n_shots)
            Q[i, state] = rng.normal(0.0, sigma, n_shots)

    sweep_values = np.linspace(-2.0, 2.0, n_sweep)
    return xr.Dataset(
        {"I": ([coord, "prepared_state", "shot_idx"], I),
         "Q": ([coord, "prepared_state", "shot_idx"], Q)},
        coords={coord: sweep_values,
                "prepared_state": [0, 1],
                "shot_idx": np.arange(n_shots)},
    )


def _make_averaged_ds(coord, **kwargs):
    """The FPGA-AVERAGED form of the same experiment: one I/Q point per prepared
    state, no ``shot_idx`` axis at all — what an ``readout_mode="average"`` probe
    returns."""
    return _make_sweep_ds(coord, **kwargs).mean("shot_idx")


class TestReadoutFidelityEstimators:

    def test_imports_match(self):
        assert ReadoutFreqFidelityEstimator is SubpkgFreq
        assert ReadoutPowerFidelityEstimator is SubpkgPower
        assert issubclass(ReadoutFreqFidelityEstimator, ReadoutFidelityEstimator)
        assert ReadoutFreqFidelityEstimator.sweep_coord == "frequency"
        assert ReadoutPowerFidelityEstimator.sweep_coord == "amp_prefactor"

    def test_freq_recovers_best_point(self):
        best_idx = 4
        ds = _make_sweep_ds("frequency", best_idx=best_idx)
        results = ReadoutFreqFidelityEstimator().extract_parameters(ds)

        assert results["sweep_coord"] == "frequency"
        assert results["fidelity"].shape == (ds.sizes["frequency"],)
        assert results["best_index"] == best_idx
        assert results["best_sweep_value"] == pytest.approx(
            ds.coords["frequency"].values[best_idx]
        )
        # SNR curve: separation / std, peaks where the blobs are most separated.
        assert results["snr"].shape == (ds.sizes["frequency"],)
        assert results["snr"][best_idx] == pytest.approx(
            results["snr"].max(), rel=0.05
        )
        # The peak must beat the sweep edges (heavy-overlap, low-fidelity points).
        assert results["best_fidelity"] > results["fidelity"][0]
        assert results["best_fidelity"] > results["fidelity"][-1]
        assert results["success"] is True

    def test_power_recovers_best_point(self):
        ds = _make_sweep_ds("amp_prefactor", best_idx=4)
        results = ReadoutPowerFidelityEstimator().extract_parameters(ds)
        assert results["sweep_coord"] == "amp_prefactor"
        assert results["best_index"] == 4
        assert results["success"] is True

    def test_metadata_drops_bulky_arrays(self):
        ds = _make_sweep_ds("frequency")
        est = ReadoutFreqFidelityEstimator()
        meta = est.extract_metadata(est.extract_parameters(ds))
        # The answer is kept...
        assert {"sweep_coord", "best_sweep_value", "best_fidelity", "success"} <= set(meta)
        # ...the heavy per-slice arrays are not.
        for bulky in ("mean", "p_outlier", "norm_res", "gaussian_norms", "direct_counts"):
            assert bulky not in meta
        # Metadata must be JSON-serialisable (via the BaseEstimator sanitiser).
        json.dumps(_json_safe(meta))

    def test_analyze_roundtrip(self, tmp_path):
        ds = _make_sweep_ds("amp_prefactor")
        est = ReadoutPowerFidelityEstimator()
        _, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "readout_power_fidelity_metadata.json").exists()
        assert (tmp_path / "readout_power_fidelity_plotdata.nc").exists()
        assert {"separation", "snr", "fidelity", "outlier", "means_on_IQ"} <= set(figs)
        # separation and std share one axes now (same IQ units), and the
        # per-component center traces are gone
        assert not {"std", "mean_distance", "mean_I", "mean_Q"} & set(figs)
        assert all(isinstance(f, plt.Figure) for f in figs.values())
        plt.close("all")

    # --- power outlier-threshold selection (deterministic, no fitting) ---------
    def _power_results(self):
        return {
            "metric": "fidelity",
            "fidelity": np.array([0.90, 0.99, 0.95]),
            # in-distribution fraction = 1 - max_k p_outlier -> [0.97, 0.90, 0.99]
            "p_outlier": np.array([[0.03, 0.0], [0.10, 0.10], [0.01, 0.01]]),
            "sweep_values": np.array([0.5, 1.0, 1.5]),
        }

    def test_power_unconstrained_picks_global_max(self):
        est = ReadoutPowerFidelityEstimator()
        assert est._select_best_index(self._power_results()) == 1  # plain argmax fidelity

    def test_power_threshold_excludes_outlier_heavy_point(self):
        est = ReadoutPowerFidelityEstimator()
        res = self._power_results()
        # threshold 0.98 -> need max p_outlier <= 0.02: only idx 0 and 2 qualify; among
        # those the higher fidelity is idx 2.
        assert est._select_best_index(res, outliers_threshold=0.98) == 2
        assert est._selection_ok(res, 2, outliers_threshold=0.98) is True
        assert est._selection_ok(res, 1, outliers_threshold=0.98) is False

    def test_power_threshold_unmet_falls_back_unsuccessful(self):
        est = ReadoutPowerFidelityEstimator()
        res = self._power_results()
        # Threshold above every in-distribution fraction (max is 0.99) -> no point
        # qualifies -> fall back to the global fidelity max, flagged not-ok.
        idx = est._select_best_index(res, outliers_threshold=0.999)
        assert idx == 1
        assert est._selection_ok(res, idx, outliers_threshold=0.999) is False


class TestAverageMethod:
    """``method="average"``: centres by averaging, no fit, no fidelity."""

    def test_registry_matches_the_method_names(self):
        from scqat.estimators.readout_fidelity.methods import METHODS

        assert set(METHODS) == {"gmm", "average"}
        assert METHODS["gmm"].requires_shots is True
        assert METHODS["average"].requires_shots is False

    def test_averaged_data_has_no_gmm_products(self):
        best_idx = 4
        ds = _make_averaged_ds("frequency", best_idx=best_idx)
        assert "shot_idx" not in ds.dims  # the acquisition really did average
        results = ReadoutFreqFidelityEstimator().extract_parameters(ds, method="average")

        assert results["method"] == "average"
        assert results["metric"] == "separation"
        assert results["best_index"] == best_idx
        assert results["success"] is True
        assert results["best_metric"] == pytest.approx(results["best_separation"])
        # nothing a Gaussian fit would have produced
        for gmm_only in ("std", "snr", "p_outlier", "norm_res", "gaussian_norms",
                         "direct_counts", "fidelity"):
            assert gmm_only not in results
        assert results["best_fidelity"] is None if "best_fidelity" in results else True

    def test_gmm_refuses_averaged_data_by_name(self):
        ds = _make_averaged_ds("frequency")
        with pytest.raises(ValueError, match="shot_idx"):
            ReadoutFreqFidelityEstimator().extract_parameters(ds)

    def test_average_refuses_the_gmm_knobs(self):
        ds = _make_averaged_ds("frequency")
        est = ReadoutFreqFidelityEstimator()
        for knob, value in (("user_std", 1.0), ("outlier_sigma", 3),
                            ("outliers_threshold", 0.98)):
            with pytest.raises(ValueError, match=knob):
                est.extract_parameters(ds, method="average", **{knob: value})

    def test_unknown_method_refused_by_name(self):
        ds = _make_averaged_ds("frequency")
        with pytest.raises(ValueError, match="mixture-of-experts"):
            ReadoutFreqFidelityEstimator().extract_parameters(
                ds, method="mixture-of-experts")

    def test_methods_agree_on_the_common_physics(self):
        """Multi-method rule 6: a method changes robustness, not physics. On
        well-separated blobs the fitted centres and the plain per-state means
        must land on the same answer."""
        ds = _make_sweep_ds("frequency", best_idx=4)
        est = ReadoutFreqFidelityEstimator()
        gmm = est.extract_parameters(ds)
        avg = est.extract_parameters(ds, method="average")

        assert avg["best_index"] == gmm["best_index"]
        assert avg["best_sweep_value"] == gmm["best_sweep_value"]
        # ...at the answer, where the blobs are RESOLVED: same centres, same
        # separation. They are compared there and not across the whole sweep
        # because the sweep edges are deliberately ~1 sigma apart, and a merged
        # pair is exactly where a mixture fit and a plain mean must differ —
        # that difference in robustness is why both methods exist.
        best = gmm["best_index"]
        np.testing.assert_allclose(avg["mean"][best], gmm["mean"][best], atol=0.15)
        assert avg["separation"][best] == pytest.approx(
            gmm["separation"][best], rel=0.1)

    def test_average_figures_are_separation_and_iq_only(self, tmp_path):
        ds = _make_averaged_ds("amp_prefactor")
        est = ReadoutPowerFidelityEstimator()
        _, figs = est.analyze(ds, output_dir=str(tmp_path), method="average")
        assert set(figs) == {"separation", "means_on_IQ"}
        assert all(isinstance(f, plt.Figure) for f in figs.values())
        plot_data = xr.open_dataset(
            tmp_path / "readout_power_fidelity_plotdata.nc")
        assert plot_data.attrs["method"] == "average"
        assert "std" not in plot_data
        plot_data.close()
        plt.close("all")

    def test_figures_render_when_every_slice_failed(self, monkeypatch):
        """A dead reduction must still leave figures: NaN curves, no crash."""
        from scqat.estimators.readout_fidelity.methods import METHODS

        def boom(*args, **kwargs):
            raise RuntimeError("no centres here")

        monkeypatch.setattr(METHODS["average"], "reduce", boom)
        ds = _make_averaged_ds("frequency")
        est = ReadoutFreqFidelityEstimator()
        results = est.extract_parameters(ds, method="average")
        assert results["failed"].all()
        assert results["success"] is False
        figs = est.generate_figures(ds, results,
                                    plot_data=est.build_plot_data(ds, results))
        assert all(isinstance(f, plt.Figure) for f in figs.values())
        plt.close("all")


class TestReadoutFreqDips:
    """ReadoutFreqFidelityEstimator dressed dip extraction and response plotting."""

    #: absolute centre the detuning axis rides on, so full_freq is realistic.
    F_CARRIER = 6.0e9

    def _make_dip_ds(self, dip0=-0.5e6, dip1=0.5e6, n_sweep=61, ql=5000.0):
        """Two notch resonances, one per prepared state, as COMPLEX S21.

        A notch is a circle in the IQ plane, so both methods can be tested on
        the same data: a magnitude-only trace (Q = 0) is a degenerate circle
        that ``method="circle"`` cannot fit, and a test built on one would pass
        only because the failure was swallowed."""
        sweep = np.linspace(-2e6, 2e6, n_sweep)
        I = np.empty((n_sweep, 2))
        Q = np.empty((n_sweep, 2))
        for state, dip in enumerate((dip0, dip1)):
            f = self.F_CARRIER + sweep
            f0 = self.F_CARRIER + dip
            s21 = 1.0 - 0.7 / (1.0 + 2j * ql * (f - f0) / f0)
            I[:, state] = np.real(s21)
            Q[:, state] = np.imag(s21)
        return xr.Dataset(
            {"I": (["frequency", "prepared_state"], I),
             "Q": (["frequency", "prepared_state"], Q)},
            coords={"frequency": sweep, "prepared_state": [0, 1]},
        )

    def test_default_dip_fit_method_is_none(self, tmp_path):
        """Default dip_fit_method is 'none': skips dip fitting and produces no response plot."""
        ds = self._make_dip_ds()
        est = ReadoutFreqFidelityEstimator()
        res = est.extract_parameters(ds, method="average")
        assert res["dip_fit_method"] == "none"
        assert res.get("detuning_dress0") is None
        assert res.get("detuning_dress1") is None
        assert res.get("chi") is None

        meta = est.extract_metadata(res)
        assert "detuning_dress0" not in meta
        assert "chi" not in meta
        assert "dip_fit_success" not in meta  # nothing was attempted

        _, figs = est.analyze(ds, output_dir=str(tmp_path), method="average")
        assert "response" not in figs
        plt.close("all")

    def test_dip_fit_method_lorentzian(self, tmp_path):
        """dip_fit_method='lorentzian' extracts dips, chi, and generates response plot."""
        dip0 = -0.6e6
        dip1 = 0.4e6
        ds = self._make_dip_ds(dip0=dip0, dip1=dip1)
        est = ReadoutFreqFidelityEstimator()
        res = est.extract_parameters(ds, method="average", dip_fit_method="lorentzian")

        assert res["dip_fit_method"] == "lorentzian"
        assert res["dip_fit_success"] is True
        # tight on purpose: the old crude fallback landed inside a loose window,
        # so a loose tolerance could not tell a real fit from a stand-in
        assert res["detuning_dress0"] == pytest.approx(dip0, abs=1e3)
        assert res["detuning_dress1"] == pytest.approx(dip1, abs=1e3)
        expected_chi = (dip0 - dip1) / 2.0
        assert res["chi"] == pytest.approx(expected_chi, abs=1e3)

        meta = est.extract_metadata(res)
        assert meta["dip_fit_method"] == "lorentzian"
        assert meta["dip_fit_success"] is True
        assert meta["detuning_dress0"] == pytest.approx(dip0, abs=1e3)
        assert meta["detuning_dress1"] == pytest.approx(dip1, abs=1e3)
        assert meta["chi"] == pytest.approx(expected_chi, abs=1e3)

        _, figs = est.analyze(ds, output_dir=str(tmp_path), method="average", dip_fit_method="lorentzian")
        assert "response" in figs
        assert isinstance(figs["response"], plt.Figure)
        plt.close("all")

    def test_dip_fit_method_circle(self, tmp_path):
        """dip_fit_method='circle' fits complex S21 using full_freq."""
        dip0 = -0.5e6
        dip1 = 0.5e6
        ds = self._make_dip_ds(dip0=dip0, dip1=dip1)
        full_freq = ds["frequency"].values + self.F_CARRIER
        ds = ds.assign_coords(full_freq=("frequency", full_freq))

        est = ReadoutFreqFidelityEstimator()
        res = est.extract_parameters(ds, method="average", dip_fit_method="circle")
        assert res["dip_fit_method"] == "circle"
        assert res["dip_fit_success"] is True
        assert res["detuning_dress0"] == pytest.approx(dip0, abs=1e3)
        assert res["detuning_dress1"] == pytest.approx(dip1, abs=1e3)
        assert res["chi"] == pytest.approx((dip0 - dip1) / 2.0, abs=1e3)
        # the absolute centres, which is the whole point of handing it full_freq
        assert res["full_freq_dress0"] == pytest.approx(self.F_CARRIER + dip0, abs=1e3)
        assert res["full_freq_dress1"] == pytest.approx(self.F_CARRIER + dip1, abs=1e3)

        _, figs = est.analyze(ds, output_dir=str(tmp_path), method="average", dip_fit_method="circle")
        assert "response" in figs
        plt.close("all")

    def test_an_unconverged_dip_is_not_reported(self, monkeypatch):
        """success=False is NOT a dip. fit_dip returns a centre either way, and
        these values become resonator facts downstream, so reading the number
        without the flag would publish an untrustworthy fit as a measurement."""
        from scqat.estimators.readout_fidelity import estimator as est_mod

        def unconverged(detuning, iq, full_freq=None, method="lorentzian", **knobs):
            return {"detuning": 1.234e6, "fwhm": 5e5, "success": False, "method": method}

        monkeypatch.setattr(est_mod, "fit_dip", unconverged)
        ds = self._make_dip_ds()
        est = ReadoutFreqFidelityEstimator()
        res = est.extract_parameters(ds, method="average", dip_fit_method="lorentzian")

        assert res["dip_fit_success"] is False
        assert res["detuning_dress0"] is None and res["detuning_dress1"] is None
        assert res["chi"] is None
        assert "fwhm_dress0" not in res            # no half-written leftovers
        # the rejected centre never leaks under any dressed-dip name
        assert not [k for k, v in res.items()
                    if k.endswith(("_dress0", "_dress1")) and v is not None]

    def test_a_raising_dip_fit_is_not_replaced_by_a_guess(self, monkeypatch):
        """A raise is the same answer as success=False: no dip, no substitute."""
        from scqat.estimators.readout_fidelity import estimator as est_mod

        def boom(*args, **kwargs):
            raise RuntimeError("no dip here")

        monkeypatch.setattr(est_mod, "fit_dip", boom)
        ds = self._make_dip_ds()
        est = ReadoutFreqFidelityEstimator()
        res = est.extract_parameters(ds, method="average", dip_fit_method="lorentzian")

        assert res["dip_fit_success"] is False
        assert res["detuning_dress0"] is None and res["detuning_dress1"] is None
        assert res["chi"] is None
        meta = est.extract_metadata(res)
        assert meta["dip_fit_success"] is False
        assert meta["detuning_dress0"] is None and meta["chi"] is None

    def test_unknown_dip_fit_method_raises(self):
        ds = self._make_dip_ds()
        est = ReadoutFreqFidelityEstimator()
        with pytest.raises(ValueError, match="dip_fit_method"):
            est.extract_parameters(ds, dip_fit_method="invalid_mode")

    def test_a_knob_from_the_other_method_raises(self):
        """baseline_order belongs to 'lorentzian' only. A caller error must
        raise rather than surface as a dip that mysteriously failed to fit."""
        ds = self._make_dip_ds()
        est = ReadoutFreqFidelityEstimator()
        with pytest.raises(ValueError, match="baseline_order"):
            est.extract_parameters(ds, method="average", dip_fit_method="circle",
                                   baseline_order=1)


