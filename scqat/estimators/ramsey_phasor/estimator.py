from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.core.figures import render_figures
from scqat.tools.lockin import lockin_phasor
from scqat.tools.fit_stretched_exp import FitStretchedExponential
from scqat.estimators.ramsey_phasor.visualization import plot_coherence, plot_phase

#: Fraction of the contrast variance the stretched exponential must explain for
#: the fit to count. lmfit reports ``success=True`` on pure noise (railing tau to
#: its lower bound and p to its upper), so its own flag is NOT a usable failure
#: signal -- the same collapse trap ``tools/ramsey_fit.py`` guards with an
#: identically-valued floor.
MIN_VAR_EXPLAINED = 0.05

#: Free parameters of the stretched exponential (a, tau, p, c).
_N_FREE = 4

#: Degrees of freedom the envelope fit must retain to mean anything.
DOF_FLOOR = 2


class RamseyPhasorEstimator(BaseEstimator):
    """Estimator for a PHASOR Ramsey experiment: the coherence envelope and the
    residual frequency error, both read off a complex lock-in over a swept
    closing-pulse phase.

    The probe plays ``x90 - idle - x90`` and, at every idle time, sweeps the
    closing pulse's FRAME through one full turn. The Ramsey fringe therefore
    lives in the **frame** axis rather than in the time axis, and a lock-in at
    one cycle per turn collapses each fringe to a single complex phasor ``z``:

    * ``abs(z)`` is the coherence CONTRAST, read directly instead of being fitted
      out of a decaying oscillation. Because the time axis no longer has to
      resolve a fringe it is free to be LOG-spaced, which is what makes the
      stretch exponent ``p`` measurable at all.
    * ``angle(z)`` is the accumulated PHASE, whose slope is the residual detuning.

    Expects an ``xarray.Dataset`` with:
        - Variable: ``'signal'`` (real, pre-discriminated) **or** complex IQ
          (``'IQdata'``, or both ``'I'`` and ``'Q'``)
        - Coordinates: ``'idle_time'`` (**seconds**) and ``'frame'``
          (closing-pulse phase, **turns**, endpoint-exclusive over one turn)

    The raw complex ``I + iQ`` is fed to the lock-in deliberately -- NOT
    ``reduced_signal``. A constant readout rotation is a global phase on ``z``,
    so it shifts ``angle(z)`` by a constant (absorbed by the slope fit's
    intercept) and leaves ``abs(z)`` untouched. No axial projection, no stored
    blob centres and no baseline subtraction are needed; the lock-in also cancels
    the readout offset exactly and rejects the ``m=2`` harmonic an over-rotated
    pi/2 pulse produces.

    Analysis chain:

    1. **Phasor per idle time by complex lock-in** at one cycle per turn
       (:func:`scqat.tools.lockin.lockin_phasor`).
    2. **Contrast** ``abs(z)`` -> stretched-exponential fit
       (:class:`scqat.tools.fit_stretched_exp.FitStretchedExponential`) ->
       ``t2_star_s`` and ``stretch_p``, gated on variance explained, on no railed
       parameter and on the degrees of freedom.
    3. **Phase** ``angle(z)`` -> prediction-corrected unwrap -> contrast-weighted
       linear fit -> ``detuning_error_hz``.

    The two halves fail INDEPENDENTLY: a phase record the unwrap cannot trust
    still yields a T2*, and a collapsed envelope still yields a detuning.

    SIGN CONVENTION: the lock-in kernel is ``exp(-2j*pi*frame)``, so a fringe
    whose phase advances produces a phasor whose angle DECREASES. The detuning is
    therefore ``-slope / (2*pi)``, not ``+slope / (2*pi)``.

    SCQO's ``update`` writes ``t2_star_s`` as the qubit's Ramsey dephasing fact
    and, when the phase fit is valid, corrects ``drive_freq_hz`` / ``f_01_hz`` by
    ``detuning_error_hz``.
    """

    estimator_name = "ramsey_phasor"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars
        )
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Phasor Ramsey analysis requires a 'signal' variable, or complex "
                "'IQdata', or both 'I' and 'Q'."
            )
        for coord in ("idle_time", "frame"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"Phasor Ramsey analysis requires a '{coord}' coordinate."
                )

    def _complex_signal(self, dataset: xr.Dataset) -> xr.DataArray:
        """The (idle_time, frame) complex signal the lock-in consumes.

        A pre-discriminated real ``signal`` is used verbatim (as a real complex);
        otherwise the raw ``I + iQ`` cloud -- the constant readout rotation is a
        global phase the contrast ignores and the slope fit absorbs.
        """
        if "signal" in dataset.data_vars:
            da = dataset["signal"].astype(complex)
        else:
            da = with_iqdata(dataset)["IQdata"]
        return da.transpose("idle_time", "frame")

    @staticmethod
    def _predictive_unwrap(
        t: np.ndarray, wrapped: np.ndarray, n_seed: int, tol: float
    ) -> Tuple[np.ndarray, int]:
        """Prefix-limited phase unwrap seeded at the dense end of the axis.

        ``np.unwrap`` over the whole record CANNOT be gated after the fact: it
        makes every step smaller than half a turn by construction, so a "reject
        where the step exceeds pi" test always passes and silently returns an
        aliased slope on a log axis. Instead the 2*pi branch of each new point is
        chosen against a running prediction from the slope fitted so far, and the
        record is CUT at the first point whose branch is ambiguous.

        Returns ``(phase, n_valid)`` where ``phase`` covers only the trusted
        leading ``n_valid`` points.
        """
        n = int(t.size)
        n_seed = max(2, int(n_seed))
        if n < n_seed + 1:
            # too short to bootstrap: such an axis is dense enough that a plain
            # unwrap is the best answer available.
            return np.unwrap(wrapped), n
        out = np.full(n, np.nan, dtype=float)
        out[:n_seed] = np.unwrap(wrapped[:n_seed])
        for i in range(n_seed, n):
            slope, icept = np.polyfit(t[:i], out[:i], 1)
            pred = slope * t[i] + icept
            cand = wrapped[i] + 2 * np.pi * np.round((pred - wrapped[i]) / (2 * np.pi))
            if abs(cand - pred) > tol:
                return out[:i], i
            out[i] = cand
        return out, n

    def _fit_contrast(
        self, t: np.ndarray, contrast: np.ndarray, fix_p, min_var_explained: float
    ) -> Dict[str, Any]:
        """Stretched-exponential fit of the coherence envelope, with the gates
        lmfit's own ``success`` flag cannot provide."""
        nan = float("nan")
        out: Dict[str, Any] = {
            "success": False,
            "t2_star_s": nan,
            "t2_star_err_s": nan,
            "stretch_p": nan,
            "stretch_p_err": nan,
            "amp": nan,
            "offset": nan,
            "var_explained": 0.0,
            "railed": False,
            "best_fit": np.full(t.size, nan, dtype=float),
        }
        if int(t.size) - _N_FREE < DOF_FLOOR or not np.isfinite(contrast).all():
            return out
        try:
            result = FitStretchedExponential(x=t, data=contrast, fix_p=fix_p).fit()
        except Exception:
            # a fitter blow-up is a failed fit, never a failed run
            return out

        best = np.asarray(result.best_fit, dtype=float)
        var = float(np.var(contrast))
        var_explained = 1.0 - float(np.var(contrast - best)) / var if var > 0 else 0.0

        tau = float(result.params["tau"].value)
        p_val = float(result.params["p"].value)
        span = float(abs(t[-1] - t[0])) or 1.0
        p_lo, p_hi = FitStretchedExponential.P_BOUNDS
        # the fitter's own guess() bounds, re-derived: a parameter sitting on one
        # is a fit that ran out of room, not one that converged.
        railed = bool(
            tau <= span * 1e-4 * 1.01
            or tau >= span * 1e3 * 0.99
            or (fix_p is None and (p_val <= p_lo * 1.01 or p_val >= p_hi * 0.99))
        )

        def _err(name: str) -> float:
            err = result.params[name].stderr
            return float(err) if err is not None else nan

        out.update({
            "success": (bool(result.success)
                        and var_explained >= min_var_explained
                        and not railed),
            "t2_star_s": tau,
            "t2_star_err_s": _err("tau"),
            "stretch_p": p_val,
            "stretch_p_err": _err("p"),
            "amp": float(result.params["a"].value),
            "offset": float(result.params["c"].value),
            "var_explained": float(var_explained),
            "railed": railed,
            "best_fit": best,
        })
        return out

    def _fit_phase(
        self,
        t: np.ndarray,
        wrapped: np.ndarray,
        weights: np.ndarray,
        n_seed: int,
        tol: float,
    ) -> Dict[str, Any]:
        """Prediction-corrected unwrap + contrast-weighted slope -> detuning.

        Never raises: a record the unwrap cannot trust degrades to NaN, with
        ``n_phase_valid`` recording how far it got.
        """
        nan = float("nan")
        n = int(t.size)
        phase_full = np.full(n, nan, dtype=float)
        fit_full = np.full(n, nan, dtype=float)
        out: Dict[str, Any] = {
            "detuning_error_hz": nan,
            "detuning_error_err_hz": nan,
            "n_phase_valid": 0,
            "phase": phase_full,
            "phase_best_fit": fit_full,
        }
        if n < 2:
            return out
        phase, n_valid = self._predictive_unwrap(t, wrapped, n_seed, tol)
        phase_full[:n_valid] = phase
        out["n_phase_valid"] = int(n_valid)
        if n_valid < 2:
            return out

        t_valid, phase_valid = t[:n_valid], phase[:n_valid]
        # a phasor's phase uncertainty goes as 1/|z|, so |z| IS the
        # inverse-variance weight -- unweighted, the slope biases high when the
        # tail has decayed into noise.
        w = np.asarray(weights[:n_valid], dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if not np.any(w > 0):
            w = None
        try:
            if n_valid >= _N_FREE and w is not None:
                coef, cov = np.polyfit(t_valid, phase_valid, 1, w=w, cov=True)
                slope_err = float(np.sqrt(abs(cov[0, 0])))
            else:
                coef = np.polyfit(t_valid, phase_valid, 1, w=w)
                slope_err = nan
        except Exception:
            return out
        slope, icept = float(coef[0]), float(coef[1])
        fit_full[:n_valid] = slope * t_valid + icept
        # kernel exp(-2j*pi*frame): an advancing fringe phase drives the phasor
        # angle NEGATIVE, hence the sign flip.
        out.update({
            "detuning_error_hz": -slope / (2 * np.pi),
            "detuning_error_err_hz": (slope_err / (2 * np.pi)
                                      if np.isfinite(slope_err) else nan),
            "phase_best_fit": fit_full,
        })
        return out

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Lock-in the fringe, then fit the coherence envelope and the phase slope.

        Kwargs -- flat and fully owned; unknown names are ignored (there is no
        multi-method surface here):
            fix_p (float | None): freeze the stretch exponent. Default ``None``
                (free), which is what a log-spaced idle axis buys you;
                ``fix_p=1.0`` recovers a plain exponential with an offset and is
                the right choice when the axis spans well under a decade.
            min_var_explained (float): fraction of the contrast variance the fit
                must explain to count as a success. Default ``0.05``. lmfit
                returns ``success=True`` on pure noise, so this gate -- not the
                lmfit flag -- is what makes a failed fit report failure.
            phase_seed_points (int): how many points at the DENSE end of the axis
                seed the prediction-corrected unwrap. Default ``5``.
            phase_tol_turns (float): branch-ambiguity threshold in TURNS; the
                unwrap stops at the first point whose nearest branch sits further
                than this from the prediction. Default ``0.25`` (a quarter turn).

        Returns a dict with:
            success, t2_star_s, t2_star_err_s, stretch_p, stretch_p_err, amp,
            offset, var_explained, railed, detuning_error_hz,
            detuning_error_err_hz, n_phase_valid, idle_time_s, contrast, phase,
            phase_wrapped, best_fit, phase_best_fit, signal.
        """
        fix_p = kwargs.get("fix_p", None)
        fix_p = None if fix_p is None else float(fix_p)
        min_var_explained = float(kwargs.get("min_var_explained", MIN_VAR_EXPLAINED))
        n_seed = int(kwargs.get("phase_seed_points", 5))
        tol = float(kwargs.get("phase_tol_turns", 0.25)) * 2 * np.pi

        cplx = self._complex_signal(dataset)
        idle_time_s = np.asarray(cplx.coords["idle_time"].values, dtype=float)
        frame = np.asarray(cplx.coords["frame"].values, dtype=float)

        # 1. complex lock-in at one cycle per turn -> one phasor per idle time
        # (the shared reduction -- ramsey_cryoscope consumes the same projection)
        z = lockin_phasor(cplx.values, frame, axis=1)
        contrast = np.abs(z)
        wrapped = np.angle(z)

        results: Dict[str, Any] = {}
        results.update(
            self._fit_contrast(idle_time_s, contrast, fix_p, min_var_explained)
        )
        results.update(
            self._fit_phase(idle_time_s, wrapped, contrast, n_seed, tol)
        )
        results.update({
            "idle_time_s": idle_time_s,
            "contrast": np.asarray(contrast, dtype=float),
            "phase_wrapped": np.asarray(wrapped, dtype=float),
            "signal": np.real(cplx.values).astype(float),
        })
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the coherence + detuning scalars; drop the traces."""
        drop = {"idle_time_s", "contrast", "phase", "phase_wrapped",
                "best_fit", "phase_best_fit", "signal"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the raw 2-D fringe map and the 1-D lock-in traces with the fit
        scalars in ``.attrs``, so both figures redraw with no recomputation.

        netCDF-safe: no complex variables -- the phasor travels as its
        contrast/phase pair, bools as int. The raw arrays are present even when
        the fit failed; fit-derived fields degrade to NaN rather than going
        missing.
        """
        idle_time_s = np.asarray(results["idle_time_s"], dtype=float)
        frame = np.asarray(dataset.coords["frame"].values, dtype=float)

        attrs = {
            "success": int(bool(results["success"])),
            "railed": int(bool(results["railed"])),
            "t2_star_s": float(results["t2_star_s"]),
            "t2_star_err_s": float(results["t2_star_err_s"]),
            "stretch_p": float(results["stretch_p"]),
            "stretch_p_err": float(results["stretch_p_err"]),
            "amp": float(results["amp"]),
            "offset": float(results["offset"]),
            "var_explained": float(results["var_explained"]),
            "detuning_error_hz": float(results["detuning_error_hz"]),
            "detuning_error_err_hz": float(results["detuning_error_err_hz"]),
            "n_phase_valid": int(results["n_phase_valid"]),
        }
        return xr.Dataset(
            {
                "signal": (("idle_time", "frame"),
                           np.asarray(results["signal"], dtype=float)),
                "contrast": ("idle_time",
                             np.asarray(results["contrast"], dtype=float)),
                "best_fit": ("idle_time",
                             np.asarray(results["best_fit"], dtype=float)),
                "phase": ("idle_time",
                          np.asarray(results["phase"], dtype=float)),
                "phase_wrapped": ("idle_time",
                                  np.asarray(results["phase_wrapped"], dtype=float)),
                "phase_best_fit": ("idle_time",
                                   np.asarray(results["phase_best_fit"], dtype=float)),
            },
            coords={"idle_time": idle_time_s, "frame": frame},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Draw the coherence fit and the phase/fringe diagnostic strictly from
        ``plot_data`` (rebuilt only when called outside ``analyze()``)."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures({
            self.estimator_name: lambda: plot_coherence(plot_data),
            "phase_detuning": lambda: plot_phase(plot_data),
        }, label=self.estimator_name)
