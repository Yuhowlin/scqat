"""Tests for the phasor Ramsey estimator.

Forward model: at each idle time the qubit's equatorial Bloch vector has length
``exp(-(t/T2)**p)`` and has precessed by ``2*pi*df*t``; sweeping the closing
pulse's frame through one turn reads that vector out as a fringe

    pop(t, frame) = 0.5 + 0.5 * env(t) * cos(2*pi*frame - phi(t))

The estimator inverts it with a lock-in at one cycle per turn: ``abs(z)``
recovers ``env`` (-> T2*, p) and ``angle(z)`` recovers ``phi`` (-> the detuning).
The IQ path adds a constant readout rotation and offset that the lock-in is
blind to by construction.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators import RamseyPhasorEstimator
from scqat.estimators.ramsey_phasor import (
    RamseyPhasorEstimator as SubpackageEstimator,
)

T2_S = 20e-6
STRETCH_P = 1.5


def _log_axis_s(min_ns=16, max_ns=200_000, num_points=60, grid_ns=4):
    """A log-spaced idle axis on the 4 ns instrument grid, as SCQO builds it.

    ``np.unique`` after snapping SHRINKS the axis, so the realized length is not
    ``num_points`` -- the estimator must never assume otherwise.
    """
    lo = max(16, int(min_ns))
    raw = np.logspace(np.log10(lo), np.log10(int(max_ns)), int(num_points))
    snapped = np.maximum(16, (np.round(raw / grid_ns) * grid_ns).astype(int))
    return np.unique(snapped).astype(float) * 1e-9


def _forward(
    t2_s=T2_S,
    p=STRETCH_P,
    detuning_hz=50e3,
    *,
    nframe=16,
    iq=False,
    theta=0.0,
    noise=0.01,
    seed=1,
    idle_s=None,
):
    """Synthesize a phasor-Ramsey dataset; returns the Dataset."""
    t = _log_axis_s() if idle_s is None else np.asarray(idle_s, dtype=float)
    frame = np.linspace(0.0, 1.0, nframe, endpoint=False)
    env = np.exp(-((t / t2_s) ** p))
    phi = 2 * np.pi * detuning_hz * t
    pop = 0.5 + 0.5 * env[:, None] * np.cos(2 * np.pi * frame[None, :] - phi[:, None])
    rng = np.random.default_rng(seed)

    if iq:
        cplx = (
            complex(0.3, -0.2)
            + pop * np.exp(1j * theta)
            + rng.normal(0, noise, pop.shape)
            + 1j * rng.normal(0, noise, pop.shape)
        )
        return xr.Dataset(
            {
                "I": (("idle_time", "frame"), cplx.real),
                "Q": (("idle_time", "frame"), cplx.imag),
            },
            coords={"idle_time": t, "frame": frame},
        )
    return xr.Dataset(
        {"signal": (("idle_time", "frame"), pop + rng.normal(0, noise, pop.shape))},
        coords={"idle_time": t, "frame": frame},
    )


def _noise_only(nframe=16, seed=7):
    t = _log_axis_s()
    frame = np.linspace(0.0, 1.0, nframe, endpoint=False)
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {"signal": (("idle_time", "frame"), 0.5 + rng.normal(0, 0.02, (t.size, nframe)))},
        coords={"idle_time": t, "frame": frame},
    )


class TestRamseyPhasorEstimator:
    def test_aggregate_and_subpackage_export_same_class(self):
        assert RamseyPhasorEstimator is SubpackageEstimator
        assert RamseyPhasorEstimator.estimator_name == "ramsey_phasor"

    def test_recovers_coherence_and_detuning(self):
        results = RamseyPhasorEstimator().extract_parameters(_forward())
        assert results["success"] is True
        assert results["t2_star_s"] == pytest.approx(T2_S, rel=0.1)
        assert results["stretch_p"] == pytest.approx(STRETCH_P, rel=0.15)
        assert results["detuning_error_hz"] == pytest.approx(50e3, rel=0.05)
        assert results["var_explained"] > 0.9
        assert results["railed"] is False

    def test_fix_p_freezes_the_exponent(self):
        results = RamseyPhasorEstimator().extract_parameters(
            _forward(p=1.0), fix_p=1.0
        )
        assert results["stretch_p"] == pytest.approx(1.0)
        assert results["t2_star_s"] == pytest.approx(T2_S, rel=0.15)

    @pytest.mark.parametrize("theta", [0.0, 1.1, -2.3])
    def test_iq_path_is_blind_to_readout_rotation(self, theta):
        """A constant readout rotation is a global phase on the phasor: it must
        move neither the contrast nor the fitted slope."""
        results = RamseyPhasorEstimator().extract_parameters(
            _forward(iq=True, theta=theta)
        )
        assert results["success"] is True
        assert results["t2_star_s"] == pytest.approx(T2_S, rel=0.1)
        assert results["detuning_error_hz"] == pytest.approx(50e3, rel=0.05)

    def test_axis_order_does_not_matter(self):
        """Estimators transpose by coordinate NAME, so callers may pass the sweep
        axes in either order."""
        dataset = _forward()
        flipped = dataset.transpose("frame", "idle_time")
        a = RamseyPhasorEstimator().extract_parameters(dataset)
        b = RamseyPhasorEstimator().extract_parameters(flipped)
        assert b["t2_star_s"] == pytest.approx(a["t2_star_s"])
        assert b["detuning_error_hz"] == pytest.approx(a["detuning_error_hz"])

    def test_detuning_sign_convention(self):
        """The kernel is exp(-2j*pi*frame), so an advancing fringe phase drives
        angle(z) NEGATIVE -- the estimator must undo that, not double it."""
        est = RamseyPhasorEstimator()
        pos = est.extract_parameters(_forward(detuning_hz=80e3))
        neg = est.extract_parameters(_forward(detuning_hz=-80e3))
        assert pos["detuning_error_hz"] == pytest.approx(80e3, rel=0.05)
        assert neg["detuning_error_hz"] == pytest.approx(-80e3, rel=0.05)

    def test_naive_unwrap_would_alias(self):
        """Pins the prediction-corrected unwrap against the obvious-but-wrong
        whole-axis np.unwrap, which aliases badly on a log axis and cannot be
        gated after the fact (unwrap makes every step < pi by construction)."""
        truth = 300e3
        dataset = _forward(detuning_hz=truth, noise=0.005)
        results = RamseyPhasorEstimator().extract_parameters(dataset)
        assert results["detuning_error_hz"] == pytest.approx(truth, rel=0.05)

        t = dataset.coords["idle_time"].values
        frame = dataset.coords["frame"].values
        kernel = np.exp(-2j * np.pi * frame)
        z = (dataset["signal"].values * kernel[None, :]).mean(axis=1)
        naive = -np.polyfit(t, np.unwrap(np.angle(z)), 1)[0] / (2 * np.pi)
        assert abs(naive - truth) > 0.5 * truth

    def test_noise_only_record_reports_failure(self):
        """lmfit returns success=True on pure noise (railing tau and p); the
        estimator's own gates are what must report the failure."""
        results = RamseyPhasorEstimator().extract_parameters(_noise_only())
        assert results["success"] is False
        assert results["var_explained"] < 0.05 or results["railed"] is True

    def test_short_record_fails_without_raising(self):
        idle = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        results = RamseyPhasorEstimator().extract_parameters(_forward(idle_s=idle))
        assert results["success"] is False
        assert np.isnan(results["t2_star_s"])

    def test_truncated_phase_does_not_cost_the_coherence(self):
        """The two halves fail independently: an unwrap cut short by an
        impossibly tight tolerance still leaves a good T2*."""
        est = RamseyPhasorEstimator()
        results = est.extract_parameters(_forward(), phase_tol_turns=1e-9)
        # cut back to the seed, so the phase is fitted from far fewer points
        assert results["n_phase_valid"] == 5
        assert results["success"] is True
        assert results["t2_star_s"] == pytest.approx(T2_S, rel=0.1)

    def test_detuning_degrades_to_nan_without_raising(self):
        """A record too short to fit any slope reports NaN and a zero count
        rather than raising or inventing a number."""
        results = RamseyPhasorEstimator().extract_parameters(
            _forward(idle_s=np.array([1e-6]))
        )
        assert np.isnan(results["detuning_error_hz"])
        assert results["n_phase_valid"] == 0
        assert results["success"] is False

    def test_check_data_requires_signal_and_coords(self):
        est = RamseyPhasorEstimator()
        good = _forward()
        with pytest.raises(ValueError, match="signal"):
            est._check_data(good.drop_vars("signal"))
        with pytest.raises(ValueError, match="idle_time"):
            est._check_data(good.rename({"idle_time": "t"}))
        with pytest.raises(ValueError, match="frame"):
            est._check_data(good.rename({"frame": "phi"}))

    def test_metadata_drops_arrays(self):
        est = RamseyPhasorEstimator()
        results = est.extract_parameters(_forward())
        metadata = est.extract_metadata(results)
        for bulky in ("idle_time_s", "contrast", "phase", "phase_wrapped",
                      "best_fit", "phase_best_fit", "signal"):
            assert bulky not in metadata
        for scalar in ("success", "t2_star_s", "stretch_p", "detuning_error_hz",
                       "n_phase_valid", "var_explained"):
            assert scalar in metadata

    def test_plot_data_layout(self):
        est = RamseyPhasorEstimator()
        dataset = _forward()
        plot_data = est.build_plot_data(dataset, est.extract_parameters(dataset))
        assert set(plot_data.coords) == {"idle_time", "frame"}
        assert plot_data["signal"].dims == ("idle_time", "frame")
        for name in ("contrast", "best_fit", "phase", "phase_wrapped",
                     "phase_best_fit"):
            assert plot_data[name].dims == ("idle_time",)
        assert plot_data.attrs["success"] == 1
        # netCDF-safe: the phasor travels as a contrast/phase pair, never complex
        assert not any(
            np.iscomplexobj(var.values) for var in plot_data.data_vars.values()
        )

    def test_analyze_roundtrips_artifacts(self, tmp_path):
        est = RamseyPhasorEstimator()
        metadata, figures = est.analyze(_forward(), output_dir=tmp_path)
        assert set(figures) == {"ramsey_phasor", "phase_detuning"}
        assert (tmp_path / "ramsey_phasor_metadata.json").exists()
        assert (tmp_path / "ramsey_phasor_plotdata.nc").exists()
        assert (tmp_path / "ramsey_phasor.png").exists()
        assert (tmp_path / "ramsey_phasor_phase_detuning.png").exists()
        reloaded = est.load_metadata(tmp_path)
        assert reloaded["estimator_name"] == "ramsey_phasor"
        assert reloaded["t2_star_s"] == pytest.approx(metadata["t2_star_s"])

    def test_figures_redraw_from_plotdata_alone(self, tmp_path):
        est = RamseyPhasorEstimator()
        dataset = _forward()
        plot_data = est.build_plot_data(dataset, est.extract_parameters(dataset))
        est.save_plot_data(plot_data, tmp_path)
        figures = est.generate_figures(None, None, plot_data=est.load_plot_data(tmp_path))
        assert set(figures) == {"ramsey_phasor", "phase_detuning"}

    def test_figures_render_on_a_failed_fit(self, tmp_path):
        """A failed fit must still produce both PNGs -- SCQO's artifact fallback
        drops ALL figures on any single plotter exception, so one log-scale
        crash on an all-NaN overlay would leave the run figure-less."""
        est = RamseyPhasorEstimator()
        dataset = _noise_only()
        results = est.extract_parameters(dataset)
        assert results["success"] is False
        plot_data = est.build_plot_data(dataset, results)
        # raw arrays survive a failed fit; fit-derived ones degrade to NaN
        assert np.isfinite(plot_data["contrast"].values).all()
        assert np.isfinite(plot_data["signal"].values).all()
        figures = est.generate_figures(None, None, plot_data=plot_data)
        assert set(figures) == {"ramsey_phasor", "phase_detuning"}
        est.save_figures(figures, tmp_path)
        assert (tmp_path / "ramsey_phasor.png").exists()
        assert (tmp_path / "ramsey_phasor_phase_detuning.png").exists()

    def test_minimum_frame_count_is_three(self):
        """The lock-in needs three frame points to separate the offset from the
        two quadratures; two must raise rather than return a plausible number."""
        est = RamseyPhasorEstimator()
        with pytest.raises(ValueError, match="at least 3 frame points"):
            est.extract_parameters(_forward(nframe=2))
        results = est.extract_parameters(_forward(nframe=3, noise=0.0))
        assert results["t2_star_s"] == pytest.approx(T2_S, rel=0.15)
