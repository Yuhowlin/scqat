"""
Parametric-Drive Decoherence Estimator
=======================================
Frequency-resolved non-Markovian decoherence of a qubit under a parametric
(flux-line) drive. This is the ``BaseEstimator`` packaging of the EP
(exceptional-point) analysis previously only available as the file-path workflow
``scqat.workflows.ep_pipeline.analyze`` (exercised by
``notebooks/EP/view_single_raw.ipynb``).

For every ``driving_frequency`` the estimator reconstructs the excited-state
population :math:`\\rho_{11}(t)`, runs the three-stage pipeline

1. Hankel pre-analysis (Matrix-Pencil seeding),
2. multi-damped-oscillation fit (seeded from the Hankel modes),
3. non-Markovian amplitude-damping fit (``FitQubitDecoherence``, :math:`\\gamma`
   seeded from Hankel mode 0),

and reports the fitted :math:`(\\gamma, \\lambda, \\Delta)` plus the
exceptional-point figure of merit :math:`8\\lambda^2/\\gamma^2` as a function of
``driving_frequency``.

Two input layouts are supported and auto-detected:

* **Tomography** (``basis`` coordinate present) — full density matrix is built
  from the X/Y/Z readouts via :func:`ep_pipeline.build_rho_dataset`; the
  ``freq_time_tomo`` node produces this.
* **:math:`\\rho_{11}`-only** (no ``basis``) — the Z-basis population is taken
  directly from the state variable and normalised
  :math:`\\rho_{11} = (\\text{state} - \\text{offset}) / \\text{scale}`; the
  ``freq_time`` node produces this.

Expected ``xarray.Dataset`` contract
-------------------------------------
The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers``).

Coordinates:
    - driving_frequency : 1-D float array — parametric drive frequency (Hz).
    - driving_time      : 1-D float array — drive duration (ns).
    - basis             : optional [0, 1, 2] X/Y/Z tomography readout index.
Data variables:
    - state (or signal / I) : the measured :math:`P(|1\\rangle)`.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.workflows.ep_pipeline import (
    build_rho_dataset,
    squeeze_singleton_dims,
    run_hankel_per_freq,
    run_mdo_per_freq,
    run_decoherence_per_freq,
    DEFAULT_RHO11_OFFSET,
    DEFAULT_RHO11_SCALE,
    DEFAULT_TAIL_FRAC,
)
from scqat.estimators.parametric_drive_decoherence.visualization import (
    plot_decoherence_params,
    plot_rho11_fits,
)

# Order matters: prefer the discriminated-state variable, then the renamed
# "signal", then the in-phase quadrature as a last resort.
_STATE_VAR_CANDIDATES = ("state", "signal", "I")

# Per-frequency scalar fields flattened from the decoherence stage.
_SCALAR_FIELDS = (
    "gamma", "gamma_err", "lambda_", "lambda_err",
    "Delta", "Delta_err", "rho_0", "rho_0_err", "chisqr",
)


class ParametricDriveDecoherenceEstimator(BaseEstimator):
    """Per-``driving_frequency`` non-Markovian decoherence fit of a
    parametrically driven qubit, reporting :math:`\\gamma`, :math:`\\lambda`,
    :math:`\\Delta` and the EP figure of merit :math:`8\\lambda^2/\\gamma^2`."""

    estimator_name = "parametric_drive_decoherence"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        for coord in ("driving_frequency", "driving_time"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"ParametricDriveDecoherenceEstimator requires a '{coord}' coordinate."
                )
        if not any(v in dataset.data_vars for v in _STATE_VAR_CANDIDATES):
            raise ValueError(
                "ParametricDriveDecoherenceEstimator requires one of the state "
                f"variables {_STATE_VAR_CANDIDATES}."
            )

    @staticmethod
    def _resolve_state_var(dataset: xr.Dataset, state_var: Optional[str]) -> str:
        if state_var is not None:
            if state_var not in dataset.data_vars:
                raise ValueError(f"state_var '{state_var}' not in dataset.")
            return state_var
        for cand in _STATE_VAR_CANDIDATES:
            if cand in dataset.data_vars:
                return cand
        raise ValueError(
            "No state variable found "
            f"(looked for {_STATE_VAR_CANDIDATES})."
        )

    @staticmethod
    def _qubit_label(dataset: xr.Dataset) -> str:
        try:
            if "qubit" in dataset.coords:
                return str(dataset.coords["qubit"].values.item())
        except Exception:
            pass
        return "parametric_drive"

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def _build_rho_ds(
        self,
        dataset: xr.Dataset,
        state_var: str,
        rho11_offset: float,
        rho11_scale: float,
    ) -> xr.Dataset:
        """Tomography → full ρ when ``basis`` is present, else ρ₁₁-only."""
        if "basis" in dataset.coords:
            ds = dataset if state_var == "state" else dataset.rename({state_var: "state"})
            return build_rho_dataset(ds, rho11_offset=rho11_offset, rho11_scale=rho11_scale)

        rho_11 = (dataset[state_var].astype(float) - rho11_offset) / rho11_scale
        rho_ds = xr.Dataset({"rho_11": rho_11}, attrs=dict(dataset.attrs))
        return squeeze_singleton_dims(rho_ds)

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Reconstruct ρ₁₁(t) and fit the decoherence model per ``driving_frequency``.

        Keyword arguments
        -----------------
        rho11_offset : float, optional
            Readout-zero subtraction applied to the Z-basis population
            (default ``ep_pipeline.DEFAULT_RHO11_OFFSET``).
        rho11_scale : float, optional
            Readout-contrast normalisation (default
            ``ep_pipeline.DEFAULT_RHO11_SCALE``).
        tail_frac : float, optional
            Fraction of the trace tail used for baseline subtraction in the
            Hankel / multi-damped-osc stages (default
            ``ep_pipeline.DEFAULT_TAIL_FRAC``).
        hankel_kwargs : dict, optional
            Overrides forwarded to ``hankel_decompose``.
        state_var : str, optional
            Force the data variable holding P(|1⟩); auto-detected otherwise.
        verbose : bool, optional
            Print per-stage progress (default False).

        Returns
        -------
        dict
            ``{driving_frequency, gamma, gamma_err, lambda_, lambda_err, Delta,
            Delta_err, rho_0, rho_0_err, chisqr, ep_metric, regime, success,
            has_tomography, n_freq, n_decoh_ok, driving_time, rho11_data,
            rho11_fit, hankel, mdo, decoh, decoh_guesses}``.
        """
        rho11_offset = float(kwargs.get("rho11_offset", DEFAULT_RHO11_OFFSET))
        rho11_scale = float(kwargs.get("rho11_scale", DEFAULT_RHO11_SCALE))
        tail_frac = float(kwargs.get("tail_frac", DEFAULT_TAIL_FRAC))
        hankel_kwargs = kwargs.get("hankel_kwargs", None)
        verbose = bool(kwargs.get("verbose", False))
        state_var = self._resolve_state_var(dataset, kwargs.get("state_var"))
        qname = self._qubit_label(dataset)
        has_tomography = "basis" in dataset.coords

        rho_ds = self._build_rho_ds(dataset, state_var, rho11_offset, rho11_scale)

        # Three-stage per-frequency pipeline (reused from ep_pipeline).
        hankel_diag = run_hankel_per_freq(
            rho_ds, tail_frac=tail_frac, hankel_kwargs=hankel_kwargs, label=qname
        )
        mdo_res = run_mdo_per_freq(rho_ds, hankel_diag, tail_frac=tail_frac, label=qname)
        decoh_res, decoh_guesses = run_decoherence_per_freq(
            rho_ds, hankel_diag, label=qname
        )

        freqs = rho_ds.coords["driving_frequency"].values.astype(float)
        t_arr = rho_ds.coords["driving_time"].values.astype(float)
        n_freq, n_time = freqs.size, t_arr.size

        # Flatten the per-frequency decoherence dict into 1-D arrays over frequency.
        scalars: Dict[str, np.ndarray] = {
            k: np.full(n_freq, np.nan) for k in _SCALAR_FIELDS
        }
        ep_metric = np.full(n_freq, np.nan)
        regime: List[str] = ["failed"] * n_freq
        success = np.zeros(n_freq, dtype=bool)
        rho11_data = np.full((n_freq, n_time), np.nan)
        rho11_fit = np.full((n_freq, n_time), np.nan)

        for i, f_val in enumerate(freqs):
            rho11_data[i, :] = rho_ds["rho_11"].sel(driving_frequency=f_val).values.astype(float)
            res = decoh_res.get(float(f_val))
            if res is None:
                continue
            for k in _SCALAR_FIELDS:
                scalars[k][i] = float(res.get(k, np.nan))
            regime[i] = res.get("regime", "failed")
            success[i] = True
            fit_curve = res.get("fit_curve")
            if fit_curve is not None:
                rho11_fit[i, :] = np.asarray(fit_curve, dtype=float)
            g, lam = scalars["gamma"][i], scalars["lambda_"][i]
            if np.isfinite(g) and g != 0.0 and np.isfinite(lam):
                ep_metric[i] = 8.0 * lam ** 2 / g ** 2

        results: Dict[str, Any] = {
            "driving_frequency": freqs,
            "driving_time": t_arr,
            "ep_metric": ep_metric,
            "regime": regime,
            "success": success,
            "has_tomography": bool(has_tomography),
            "n_freq": int(n_freq),
            "n_decoh_ok": int(success.sum()),
            # bulky intermediates (kept for plot data / debugging, dropped from metadata)
            "rho11_data": rho11_data,
            "rho11_fit": rho11_fit,
            "hankel": hankel_diag,
            "mdo": mdo_res,
            "decoh": decoh_res,
            "decoh_guesses": decoh_guesses,
        }
        results.update(scalars)
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the per-frequency scalar arrays; drop the 2-D maps and raw
        stage dictionaries (which carry arrays / fit objects)."""
        drop = {
            "driving_time", "rho11_data", "rho11_fit",
            "hankel", "mdo", "decoh", "decoh_guesses",
        }
        return {k: v for k, v in results.items() if k not in drop}

    # ------------------------------------------------------------------
    # Plot data
    # ------------------------------------------------------------------
    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Self-contained Dataset: per-frequency fit scalars (over
        ``driving_frequency``) plus the ρ₁₁ data and fit maps (over
        ``driving_frequency`` × ``driving_time``)."""
        freqs = np.asarray(results["driving_frequency"], dtype=float)
        t_arr = np.asarray(results["driving_time"], dtype=float)

        data_vars: Dict[str, Any] = {
            "rho11_data": (("driving_frequency", "driving_time"),
                           np.asarray(results["rho11_data"], dtype=float)),
            "rho11_fit": (("driving_frequency", "driving_time"),
                          np.asarray(results["rho11_fit"], dtype=float)),
            "ep_metric": ("driving_frequency", np.asarray(results["ep_metric"], dtype=float)),
            "success": ("driving_frequency", np.asarray(results["success"], dtype=int)),
        }
        for k in _SCALAR_FIELDS:
            data_vars[k] = ("driving_frequency", np.asarray(results[k], dtype=float))

        return xr.Dataset(
            data_vars,
            coords={"driving_frequency": freqs, "driving_time": t_arr},
            attrs={
                "has_tomography": int(bool(results["has_tomography"])),
                "n_freq": int(results["n_freq"]),
                "n_decoh_ok": int(results["n_decoh_ok"]),
            },
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Two figures, drawn entirely from ``plot_data``:
        ``decoherence_params`` (γ, λ, |Δ|, 8λ²/γ² vs driving_frequency) and
        ``rho11_fits`` (ρ₁₁(t) data + fit, coloured by driving_frequency).

        Built through :func:`render_figures` so the two are INDEPENDENT: the
        raw-carrying ``rho11_fits`` must survive a crash in the pure-fit
        ``decoherence_params`` panel. A consumer that saves artifacts (SCQO)
        drops ALL figures on any single plotter exception, so without the
        isolation one broken panel would cost the run every PNG."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures(
            {
                "decoherence_params": lambda: plot_decoherence_params(plot_data),
                "rho11_fits": lambda: plot_rho11_fits(plot_data),
            },
            label=self.estimator_name,
        )
