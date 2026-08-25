from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.core.figures import render_figures
from scqat.tools.lockin import lockin_phasor
from scqat.tools.step_response_fit import fit_step_response, mpm_tau_seeds
from scqat.estimators.ramsey_cryoscope.visualization import plot_step_response, plot_phase_freq


class RamseyCryoscopeEstimator(BaseEstimator):
    """Estimator for a RAMSEY cryoscope experiment: reconstruct the flux line's step
    response and fit it to a sum of exponentials, yielding the predistortion
    tap ``(amplitude, tau)`` components.

    The probe plays a flux pulse of swept DURATION inside a Ramsey sequence and
    sweeps the closing pulse's FRAME (a full turn of phase tomography); the
    qubit's flux-dependent detuning imprints a phase that grows with the
    accumulated flux. Expects an ``xarray.Dataset`` with:
        - Variable: ``'signal'`` (real, pre-discriminated) **or** complex IQ
          (``'IQdata'``, or both ``'I'`` and ``'Q'``)
        - Coordinates: ``'duration'`` (flux-pulse duration, **seconds**) and
          ``'frame'`` (closing-pulse phase, **turns**)

    Analysis chain (no ``qualibration_libs`` dependency — reimplemented here):

    1. **Phase per duration by complex lock-in** at exactly one cycle per turn:
       ``phase[d] = angle(mean_frame(C[d] * exp(-2j*pi*frame)))``. For the known
       frequency this is the maximum-likelihood phase, needs no per-slice fit,
       and — for complex IQ — uses the full IQ contrast regardless of the
       readout rotation (the constant rotation is a global phase offset that
       cancels in the unwrap + derivative below).
    2. **Unwrap** the phase along duration.
    3. **Savitzky-Golay derivative** of ``phase / (2*pi)`` → the detuning
       ``freq_hz(t)`` in Hz (``delta`` is the duration step in seconds).
    4. **Flux ∝ sqrt(|freq_hz|)** (quadratic arch near the sweet spot),
       **normalized by the settled tail** so the step response settles to ~1 —
       so the recorded taps are relative and no flux-frequency curvature
       calibration is needed.
    5. **Multi-exponential fit** (:func:`scqat.tools.step_response_fit.fit_step_response`)
       → the ``(amp, tau_s)`` components.

    SCQO's ``update`` writes the components as the flux channel's paired facts
    ``distortion_amp`` (= ``amp / a_dc``) and ``distortion_tau_s`` (seconds).
    """

    estimator_name = "ramsey_cryoscope"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars
        )
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Cryoscope analysis requires a 'signal' variable, or complex "
                "'IQdata', or both 'I' and 'Q'."
            )
        for coord in ("duration", "frame"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"Cryoscope analysis requires a '{coord}' coordinate."
                )

    def _complex_signal(self, dataset: xr.Dataset) -> xr.DataArray:
        """The (duration, frame) complex signal the lock-in consumes.

        A pre-discriminated real ``signal`` is used verbatim (as a real complex);
        otherwise the raw ``I + iQ`` cloud — the constant readout rotation is a
        global phase that the downstream unwrap + derivative removes.
        """
        if "signal" in dataset.data_vars:
            da = dataset["signal"].astype(complex)
        else:
            da = with_iqdata(dataset)["IQdata"]
        return da.transpose("duration", "frame")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Reconstruct the step response and fit its exponential components.

        Kwargs — flat and fully owned; unknown names are ignored (there is no
        multi-method surface here):
            seed_method ("mpm" | "fractions"): how the exponential fit is
                seeded. Default ``"mpm"`` — Hankel-SVD Matrix Pencil picks the
                model order + tau seeds from the data (the uniform 1 ns axis is
                exactly MPM's requirement), then one bounded JOINT fit; the
                sequential ``"fractions"`` fit always runs too and the better
                residual wins, so the default is never worse than the old
                behavior. ``"fractions"`` skips MPM entirely.
            tau_seeds (sequence[float] | None): EXPLICIT tau seeds in seconds —
                PRIOR KNOWLEDGE (e.g. the previous run's taps). When given they
                take precedence over MPM: the seeded joint fit is the candidate
                (racing the fractions fit as usual, better residual wins) and
                the winner reports ``seed_method="seeded"``.
            start_fractions (sequence[float]): DESCENDING 0-1 fractions where each
                exponential component's fit starts (slowest first). Default
                ``(0.5, 0.01)``. Used by the fractions path (and the fallback).
            stable_tail_s (float): width of the settled-tail window used to
                normalize the step response, seconds. Default ``20e-9``.
            sg_window, sg_order (int): Savitzky-Golay window / polynomial order
                for the phase derivative. Defaults ``3`` / ``2``.
            a_dc (float | None): pin the settled level of the exponential fit.
                Default ``1.0`` (the step response is normalized so the tail is
                ~1); pass ``None`` to estimate it from the flattest tail.

        Returns a dict with:
            success, component_amps, component_taus_s, a_dc, rms_residual,
            n_components, start_fractions, stable_tail_s, seed_method,
            mpm_n_modes, mpm_oscillatory,
            duration_s, phase, freq_hz, step_response, best_fit, signal.
        """
        seed_method = str(kwargs.get("seed_method", "mpm"))
        tau_seeds = [float(x) for x in (kwargs.get("tau_seeds") or [])]
        start_fractions = list(kwargs.get("start_fractions", (0.5, 0.01)))
        stable_tail_s = float(kwargs.get("stable_tail_s", 20e-9))
        sg_window = int(kwargs.get("sg_window", 3))
        sg_order = int(kwargs.get("sg_order", 2))
        a_dc = kwargs.get("a_dc", 1.0)

        cplx = self._complex_signal(dataset)
        duration_s = np.asarray(cplx.coords["duration"].values, dtype=float)
        frame = np.asarray(cplx.coords["frame"].values, dtype=float)

        # 1. complex lock-in at one cycle per turn -> phase per duration
        # (the shared reduction — ramsey_phasor consumes the same projection)
        proj = lockin_phasor(cplx.values, frame, axis=1)
        phase = np.unwrap(np.angle(proj))

        # 2-3. Savitzky-Golay derivative of phase/(2*pi) -> detuning in Hz.
        # savgol needs an odd window no longer than the record.
        window = min(sg_window if sg_window % 2 else sg_window + 1,
                     duration_s.size - (1 - duration_s.size % 2))
        window = max(window, sg_order + 1 + (sg_order % 2 == 0))
        dt_s = float(np.mean(np.diff(duration_s)))
        freq_hz = savgol_filter(
            phase / (2 * np.pi), window, sg_order, deriv=1, delta=dt_s,
        )

        # 4. flux ∝ sqrt(|detuning|), normalized by the settled tail
        flux = np.sqrt(np.abs(freq_hz))
        tail = duration_s >= (duration_s[-1] - stable_tail_s)
        tail_mean = float(np.mean(flux[tail])) if np.any(tail) else float("nan")
        step_response = (flux / tail_mean if tail_mean and np.isfinite(tail_mean)
                         else np.full_like(flux, np.nan))

        # 5. multi-exponential fit -> predistortion taps. MPM-first: the
        # Matrix Pencil picks the model order + tau seeds from the data (no
        # fraction heuristics, no sequential-subtraction bias), then ONE
        # bounded joint fit. The sequential start-fractions fit ALWAYS runs
        # too, and the better residual wins — so the default is never worse
        # than the fractions-only behavior (on clean well-separated data the
        # fractions path tends to win; on a fast tap beneath a slow one — the
        # real-hardware regime — MPM does).
        used_method, mpm_n_modes, mpm_oscillatory = "fractions", 0, 0
        candidate, candidate_label = None, ""
        if tau_seeds:
            # explicit prior-knowledge seeds take precedence over MPM
            try:
                candidate = fit_step_response(
                    duration_s, step_response, start_fractions,
                    a_dc=a_dc, tau_seeds=tau_seeds,
                )
                candidate_label = "seeded"
                if not candidate["success"]:
                    candidate = None
            except (ValueError, RuntimeError):
                pass  # bad seeds -> fractions only
        elif seed_method == "mpm":
            try:
                seeds = mpm_tau_seeds(duration_s, step_response, a_dc=a_dc or 1.0)
                mpm_n_modes = int(seeds["n_modes"])
                mpm_oscillatory = int(seeds["oscillatory"])
                if seeds["taus"]:
                    candidate = fit_step_response(
                        duration_s, step_response, start_fractions,
                        a_dc=a_dc, tau_seeds=seeds["taus"],
                    )
                    candidate_label = "mpm"
                    if not candidate["success"]:
                        candidate = None
            except (ValueError, RuntimeError):
                pass  # non-uniform axis / too few points -> fractions only
        fit = fit_step_response(
            duration_s, step_response, start_fractions, a_dc=a_dc,
        )
        if candidate is not None and (
            not fit["success"] or candidate["rms"] < fit["rms"]
        ):
            fit, used_method = candidate, candidate_label
        amps = [amp for amp, _ in fit["components"]]
        taus = [tau for _, tau in fit["components"]]

        return {
            "success": bool(fit["success"]),
            "component_amps": amps,
            "component_taus_s": taus,
            "a_dc": float(fit["a_dc"]),
            "rms_residual": float(fit["rms"]),
            "n_components": len(fit["components"]),
            "start_fractions": [float(f) for f in fit["best_fractions"]],
            "seed_method": used_method,
            "mpm_n_modes": mpm_n_modes,
            "mpm_oscillatory": mpm_oscillatory,
            "stable_tail_s": stable_tail_s,
            "duration_s": duration_s,
            "phase": phase,
            "freq_hz": freq_hz,
            "step_response": step_response,
            "best_fit": np.asarray(fit["best_fit"], dtype=float),
            "signal": np.real(cplx.values).astype(float),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the tap components + fit scalars; drop the diagnostic arrays."""
        drop = {"duration_s", "phase", "freq_hz", "step_response",
                "best_fit", "signal"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the reconstructed 1-D traces (over ``duration``) and the raw
        2-D fringe map (over ``duration`` x ``frame``) with the tap components in
        ``.attrs`` (netCDF-safe: ``success`` as int, components as 1-D arrays),
        so both figures redraw with no recomputation."""
        duration_s = np.asarray(results["duration_s"], dtype=float)
        frame = np.asarray(dataset.coords["frame"].values, dtype=float)

        attrs = {
            "success": int(bool(results["success"])),
            "a_dc": float(results["a_dc"]),
            "rms_residual": float(results["rms_residual"]),
            "n_components": int(results["n_components"]),
            "fit_component_amps": np.asarray(results["component_amps"], dtype=float),
            "fit_component_taus_s": np.asarray(results["component_taus_s"], dtype=float),
            "stable_tail_s": float(results["stable_tail_s"]),
        }
        return xr.Dataset(
            {
                "signal": (("duration", "frame"), np.asarray(results["signal"], dtype=float)),
                "phase": ("duration", np.asarray(results["phase"], dtype=float)),
                "freq_hz": ("duration", np.asarray(results["freq_hz"], dtype=float)),
                "step_response": ("duration", np.asarray(results["step_response"], dtype=float)),
                "best_fit": ("duration", np.asarray(results["best_fit"], dtype=float)),
            },
            coords={"duration": duration_s, "frame": frame},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Draw the step-response fit and the phase/detuning diagnostic strictly
        from ``plot_data`` (rebuilt only when called outside ``analyze()``)."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures({
            self.estimator_name: lambda: plot_step_response(plot_data),
            "phase_freq": lambda: plot_phase_freq(plot_data),
        }, label=self.estimator_name)
