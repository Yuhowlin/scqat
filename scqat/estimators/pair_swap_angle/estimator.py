"""The exchange ANGLE a partial swap applies, as a function of its angle knob.

Excite one member of a pair, apply the SAME swap ``N`` times, and read both
members out jointly -- repeated for each value of the knob that sets the angle
(the coupler flux; ``TUTORIAL.md`` section 12: a member's own flux amplitude is
the RESONANCE knob and the duration is baked, so the coupler is what tunes
``J_eff`` and hence ``theta = J_eff * t_p``). At each knob value the transfer
oscillates in ``N`` with period ``pi / theta``, so fitting a cosine per knob row
turns the map into the calibration curve ``theta(knob)``: the angle you actually
get for the volts you actually set.

This is the reading ``qc_n_swap_amp`` does NOT make. That experiment sweeps the
member's flux amplitude to FIND the resonance by error amplification and only
summarises where transfer peaks; the angle is left to be counted off the figure
by eye. Here the swept axis is the angle knob and the model is the cosine in
``N``, so the extracted quantity is an angle in radians.

The per-trace fit is :func:`scqat.tools.fit_cosine.fit_swap_oscillation`, shared
with ``swap_oscillation`` -- a numerical routine, reused freely; the MODEL claim
(that this map is theta(knob)) is what this estimator owns.

Record-only: the SUCCESS / ``min_transfer`` verdict stays in SCQO, and nothing is
written to the device.

Dataset contract (the unified readout schema's joint form):
  vars   : ``joint_population`` -- dims ``(joint_state, coupler_flux_v,
           swap_count)`` in any order; ``joint_state`` labels
           ``"00"/"01"/"10"/"11"`` (leftmost digit = the HIGH member)
  coords : ``joint_state`` / ``coupler_flux_v`` (V) / ``swap_count``
           (dimensionless N)
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) selects the transfer partner;
           ``flux_side`` / ``high_name`` / ``low_name`` label the figure;
           ``target_theta_rad`` (float | None) is the angle to solve the curve
           for.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators._pair_swap_maps import (
    pair_swap_plot_data,
    plot_pair_swap_map,
    summarize_pair_swap,
)
from scqat.estimators.pair_swap_angle.visualization import plot_angle_calibration
from scqat.tools.fit_cosine import fit_swap_oscillation

AXIS0 = "coupler_flux_v"
AXIS1 = "swap_count"

#: figure keys. ``save_figures`` prefixes the estimator name unless the key IS
#: it, so these land as ``pair_swap_angle.png`` and ``pair_swap_angle_angle.png``.
FIG_MAP = "pair_swap_angle"
FIG_ANGLE = "angle"


def _transfer_marginal(plot_data: xr.Dataset, drive_side: str) -> np.ndarray:
    """P(partner excited) over ``(coupler_flux_v, swap_count)``.

    The partner is the member that was NOT driven, so its marginal is the
    transfer curve. Traced out of the four joint basis maps that
    ``pair_swap_plot_data`` already projected -- the public output of the shared
    module, so no private helper is reached into.
    """
    p11 = np.asarray(plot_data["p11"].values, dtype=float)
    partner = "p10" if drive_side == "low" else "p01"
    return np.asarray(plot_data[partner].values, dtype=float) + p11


def _as_row(values, size: int, dtype):
    """A per-knob column of the right length, NaN/0-filled when absent.

    ``build_plot_data`` may be called standalone (the replot path) with a results
    dict that never held the fit, so a missing curve degrades rather than raising
    -- rule 1 of "raw data must always be plottable".
    """
    if values is None:
        return np.full(size, np.nan if dtype is float else 0, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _solve_for_angle(knob, theta, ok, target: float) -> Dict[str, float]:
    """The knob value giving ``target`` radians, by linear interpolation.

    Only rows whose fit SUCCEEDED are candidates: a failed row carries a NaN
    theta and interpolating through it would invent an angle. When the target is
    bracketed by two successive good rows the answer is interpolated between
    them; otherwise it degrades to the nearest good row, which is honest about a
    target the sweep does not reach. An empty candidate set gives NaN, never a
    raise.
    """
    good = np.flatnonzero(ok)
    if good.size == 0 or not np.isfinite(target):
        return {"best_coupler_flux_v": float("nan"),
                "best_theta_rad": float("nan"),
                "best_is_interpolated": 0}
    k, t = knob[good], theta[good]
    # Walk consecutive good rows looking for a bracket. The curve need not be
    # monotonic overall (J_eff can turn over), so the FIRST bracket is taken
    # rather than assuming a global inverse exists.
    for i in range(k.size - 1):
        lo, hi = t[i], t[i + 1]
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            frac = (target - lo) / (hi - lo)
            return {"best_coupler_flux_v": float(k[i] + frac * (k[i + 1] - k[i])),
                    "best_theta_rad": float(target),
                    "best_is_interpolated": 1}
    j = int(np.argmin(np.abs(t - target)))
    return {"best_coupler_flux_v": float(k[j]),
            "best_theta_rad": float(t[j]),
            "best_is_interpolated": 0}


class PairSwapAngleEstimator(BaseEstimator):
    """Fit the swap angle per angle-knob value and return the theta(knob) curve."""

    estimator_name = "pair_swap_angle"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError(
                "pair_swap_angle estimator requires the joint_population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in ("joint_state", AXIS0, AXIS1):
            if axis not in dataset.coords:
                raise ValueError(
                    f"pair_swap_angle estimator requires a {axis!r} coordinate"
                )

    def extract_parameters(
        self, dataset: xr.Dataset, drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None,
        target_theta_rad: Optional[float] = None, **kwargs
    ) -> Dict[str, Any]:
        # The map summary (where transfer peaks, the marginal ranges) is the same
        # reduction every pair-swap map reports; only the angle fit below is new.
        results = summarize_pair_swap(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        projected = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        transfer = _transfer_marginal(projected, drive_side)      # (knob, N)
        knob = np.asarray(dataset[AXIS0].values, dtype=float)
        counts = np.asarray(dataset[AXIS1].values, dtype=float)

        # One cosine fit per knob row. A row that cannot be fitted degrades to
        # NaN with success False -- the map still draws.
        theta = np.full(knob.size, np.nan)
        r_squared = np.full(knob.size, np.nan)
        period = np.full(knob.size, np.nan)
        ok = np.zeros(knob.size, dtype=bool)
        best_fit = np.full(transfer.shape, np.nan)
        for i in range(knob.size):
            fit = fit_swap_oscillation(counts, transfer[i])
            theta[i] = fit["theta_rad"]
            r_squared[i] = fit["r_squared"]
            period[i] = fit["swap_period"]
            ok[i] = bool(fit["success"])
            best_fit[i] = fit["best_fit"]

        target = (float(target_theta_rad) if target_theta_rad is not None
                  else float("nan"))
        results.update({
            "n_theta_ok": int(ok.sum()),
            "target_theta_rad": target,
            "theta_min_rad": float(np.nanmin(theta[ok])) if ok.any() else float("nan"),
            "theta_max_rad": float(np.nanmax(theta[ok])) if ok.any() else float("nan"),
            **_solve_for_angle(knob, theta, ok, target),
        })
        # The curve itself, as plain lists so the metadata JSON stays portable.
        results["theta_rad"] = [float(v) for v in theta]
        results["theta_success"] = [int(v) for v in ok]
        results["theta_r_squared"] = [float(v) for v in r_squared]
        results["swap_period"] = [float(v) for v in period]
        # underscore-prefixed: bulky arrays kept for build_plot_data only
        results["_transfer"] = transfer
        results["_best_fit"] = best_fit
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the bulky per-row arrays; the theta curve is small and stays."""
        return {k: v for k, v in results.items() if not k.startswith("_")}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None,
        **kwargs
    ) -> Optional[xr.Dataset]:
        out = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        dims = (AXIS0, AXIS1)
        transfer = results.get("_transfer")
        if transfer is None:
            transfer = _transfer_marginal(out, drive_side)
        out["transfer"] = (dims, np.asarray(transfer, dtype=float))
        best_fit = results.get("_best_fit")
        if best_fit is None:
            best_fit = np.full(np.shape(transfer), np.nan)
        # The fit degrades to NaN, never to absent, so the raw transfer map still
        # draws when every row failed.
        out["transfer_fit"] = (dims, np.asarray(best_fit, dtype=float))
        n_knob = int(out.sizes[AXIS0])
        out["theta_rad"] = ((AXIS0,), _as_row(results.get("theta_rad"), n_knob, float))
        out["theta_success"] = ((AXIS0,),
                                _as_row(results.get("theta_success"), n_knob, int))
        out["theta_r_squared"] = ((AXIS0,),
                                  _as_row(results.get("theta_r_squared"), n_knob, float))
        out.attrs.update({
            "target_theta_rad": float(results.get("target_theta_rad", float("nan"))),
            "best_coupler_flux_v": float(
                results.get("best_coupler_flux_v", float("nan"))),
            "best_theta_rad": float(results.get("best_theta_rad", float("nan"))),
            "best_is_interpolated": int(results.get("best_is_interpolated", 0)),
            "n_theta_ok": int(results.get("n_theta_ok", 0)),
        })
        return out

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(
                dataset, results,
                drive_side=kwargs.get("drive_side", "low"),
                flux_side=kwargs.get("flux_side"),
                high_name=kwargs.get("high_name"),
                low_name=kwargs.get("low_name"),
            )
        return render_figures(
            {
                FIG_MAP: lambda: plot_pair_swap_map(plot_data),
                FIG_ANGLE: lambda: plot_angle_calibration(plot_data),
            },
            label=self.estimator_name,
        )
