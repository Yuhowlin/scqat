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

    def _make_dip_ds(self, dip0=-0.5e6, dip1=0.5e6, n_sweep=31):
        sweep = np.linspace(-2e6, 2e6, n_sweep)
        gamma = 0.3e6
        I = np.empty((n_sweep, 2))
        Q = np.zeros((n_sweep, 2))
        for i, f in enumerate(sweep):
            I[i, 0] = 1.0 - 0.7 / (1.0 + ((f - dip0) / gamma) ** 2)
            I[i, 1] = 1.0 - 0.7 / (1.0 + ((f - dip1) / gamma) ** 2)
        return xr.Dataset(
            {"I": (["frequency", "prepared_state"], I),
             "Q": (["frequency", "prepared_state"], Q)},
            coords={"frequency": sweep, "prepared_state": [0, 1]},
        )

    def test_extract_dips_and_chi(self):
        dip0 = -0.6e6
        dip1 = 0.4e6
        ds = self._make_dip_ds(dip0=dip0, dip1=dip1)
        est = ReadoutFreqFidelityEstimator()
        res = est.extract_parameters(ds, method="average")

        assert res["detuning_dress0"] == pytest.approx(dip0, abs=5e4)
        assert res["detuning_dress1"] == pytest.approx(dip1, abs=5e4)
        expected_chi = (dip0 - dip1) / 2.0
        assert res["chi"] == pytest.approx(expected_chi, abs=5e4)

        # Metadata carries the scalar dip results
        meta = est.extract_metadata(res)
        assert meta["detuning_dress0"] == pytest.approx(dip0, abs=5e4)
        assert meta["detuning_dress1"] == pytest.approx(dip1, abs=5e4)
        assert meta["chi"] == pytest.approx(expected_chi, abs=5e4)

    def test_response_figure_generated(self, tmp_path):
        ds = self._make_dip_ds()
        est = ReadoutFreqFidelityEstimator()
        _, figs = est.analyze(ds, output_dir=str(tmp_path), method="average")
        assert "response" in figs
        assert isinstance(figs["response"], plt.Figure)
        plt.close("all")

