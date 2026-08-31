from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_cosine import fit_swap_oscillation
from scqat.estimators.swap_oscillation.visualization import plot_rounds_fit


class SwapOscillationEstimator(BaseEstimator):
    """
    Analyzes a swap-chain (N-swap) sweep to extract the swap-oscillation frequency.

    Expects an xarray.Dataset with:
        - Variable: 'signal'
        - Coordinate: 'round'  (integer number of swaps applied; N=0 — the no-swap
          baseline — is allowed)

    A coherent (partial) swap exchanges population between the pair swap by swap, so
    the measured population follows a non-decaying cosine in N. Fits
    ``a*cos(2*pi*f*x + phi) + c`` (via :class:`FitCosine`) and reports ``f`` — the
    oscillation frequency in cycles per swap — and ``swap_period = 1/f`` — the number
    of swaps per full population cycle.

    Note on the full-swap limit: a *full* iSWAP oscillates at exactly the Nyquist
    limit of the integer-N sweep (f = 0.5 for a step of 1), where ``a`` and ``phi``
    are individually degenerate (``cos(pi*N + phi) = cos(pi*N)*cos(phi)`` at integer
    N); ``f`` itself is still well determined.
    """

    estimator_name = "swap_oscillation"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset:
            raise ValueError("Swap-oscillation analysis requires a 'signal' variable in the dataset.")
        if "round" not in dataset.coords:
            raise ValueError("Swap-oscillation analysis requires a 'round' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the swap oscillation and extract its frequency (cycles per swap).

        The fit itself is the shared per-trace reduction
        :func:`scqat.tools.fit_cosine.fit_swap_oscillation` (two phase seeds, a
        contrast gate and an R^2 gate), so this estimator and the per-row map fit
        in ``pair_swap_angle`` cannot drift apart.

        Returns a dict with:
            a, f, phi, c, theta_rad, swap_period, r_squared, success, best_fit,
            round_dense, best_fit_dense, fit_report.
        """
        signal = dataset["signal"].squeeze()
        rounds = np.asarray(dataset.coords["round"].values, dtype=float)
        fit = fit_swap_oscillation(rounds, np.asarray(signal.values, dtype=float))
        # `round` is this estimator's axis name for the shared helper's generic `x`.
        fit["round_dense"] = fit.pop("x_dense")
        return fit

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the fit parameters and swap period; drop the diagnostic arrays."""
        drop = {"best_fit", "round_dense", "best_fit_dense", "fit_report"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the raw signal + best-fit curve over ``round``; the fit parameters and
        ``swap_period`` live in ``.attrs`` so the figure needs no recomputation.
        """
        rounds = np.asarray(dataset.coords["round"].values, dtype=float)
        signal = np.asarray(dataset["signal"].squeeze().values, dtype=float)
        best_fit = np.asarray(results["best_fit"], dtype=float)

        attrs = {
            "a": float(results["a"]),
            "f": float(results["f"]),
            "phi": float(results["phi"]),
            "c": float(results["c"]),
            "swap_period": float(results["swap_period"]),
            "success": int(bool(results["success"])),
        }

        return xr.Dataset(
            {
                "signal": ("round", signal),
                "best_fit": ("round", best_fit),
                "best_fit_dense": ("round_dense", np.asarray(results["best_fit_dense"], dtype=float)),
            },
            coords={
                "round": rounds,
                "round_dense": np.asarray(results["round_dense"], dtype=float),
            },
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the rounds-fit plot, drawing strictly from ``plot_data`` so the
        figure stays reconstructable downstream; rebuild it only when called outside
        ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"rounds": plot_rounds_fit(plot_data)}
