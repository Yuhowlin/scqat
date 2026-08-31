"""Angle-calibration plotting for ``pair_swap_angle``.

Consumes the **plot_data** Dataset built by
``PairSwapAngleEstimator.build_plot_data`` and draws without any recalculation.

plot_data layout
----------------
coords : ``coupler_flux_v``, ``swap_count``
vars   : ``p00`` / ``p01`` / ``p10`` / ``p11`` (the joint basis maps, drawn by
         the shared ``plot_pair_swap_map``), ``transfer`` and ``transfer_fit``
         over ``(coupler_flux_v, swap_count)``, and ``theta_rad`` /
         ``theta_success`` / ``theta_r_squared`` over ``coupler_flux_v``
attrs  : ``target_theta_rad``, ``best_coupler_flux_v``, ``best_theta_rad``,
         ``best_is_interpolated``, ``n_theta_ok`` (plus the shared map attrs)
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

AXIS0 = "coupler_flux_v"
AXIS1 = "swap_count"

#: reference angles worth seeing on the theta axis, as (radians, label).
_LANDMARKS = (
    (np.pi / 2, r"$\pi/2$  (iSWAP)"),
    (np.pi / 4, r"$\pi/4$  ($\sqrt{\mathrm{iSWAP}}$)"),
)


def plot_angle_calibration(plot_data: xr.Dataset) -> plt.Figure:
    """Transfer-vs-N map beside the fitted angle-vs-knob calibration curve.

    Left: the raw transfer marginal over (angle knob, swap count) -- the picture
    the fit is made from, where the oscillation period visibly shortens as the
    angle grows. Right: the fitted ``theta`` per knob value, with the requested
    angle and the knob value that delivers it marked.

    The raw map is drawn UNCONDITIONALLY; every fit-derived overlay is guarded,
    so a run in which every row failed still produces this figure.
    """
    fig, (ax_map, ax_theta) = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True
    )

    knob = np.asarray(plot_data[AXIS0].values, dtype=float)
    counts = np.asarray(plot_data[AXIS1].values, dtype=float)
    transfer = np.asarray(plot_data["transfer"].values, dtype=float)  # (knob, N)

    # --- left: the raw transfer map (always drawn) -----------------------
    mesh = ax_map.pcolormesh(
        *np.meshgrid(knob, counts), transfer.T,
        shading="auto", cmap="viridis", vmin=0.0, vmax=1.0,
    )
    fig.colorbar(mesh, ax=ax_map, label="transfer population")
    ax_map.set_xlabel("coupler flux (V)")
    ax_map.set_ylabel("swap count N")
    ax_map.set_title("transfer vs (angle knob, N)")

    # --- right: the fitted angle curve (guarded) -------------------------
    theta = np.asarray(_get(plot_data, "theta_rad", knob.size), dtype=float)
    ok = np.asarray(_get(plot_data, "theta_success", knob.size), dtype=int).astype(bool)

    if np.isfinite(theta).any():
        ax_theta.plot(knob[ok], theta[ok], "o-", color="tab:blue",
                      label="fitted (converged)", markersize=5)
        if (~ok).any():
            ax_theta.plot(knob[~ok], theta[~ok], "o", mfc="none", color="tab:red",
                          label="fit rejected", markersize=5)
    else:
        ax_theta.text(0.5, 0.5, "no angle could be fitted\n(raw map at left)",
                      transform=ax_theta.transAxes, ha="center", va="center",
                      fontsize=12, color="tab:red")

    for value, label in _LANDMARKS:
        ax_theta.axhline(value, color="0.75", lw=1, ls=":")
        ax_theta.annotate(label, xy=(knob[0], value), fontsize=9, color="0.45",
                          va="bottom", ha="left")

    target = float(plot_data.attrs.get("target_theta_rad", float("nan")))
    best_v = float(plot_data.attrs.get("best_coupler_flux_v", float("nan")))
    if np.isfinite(target):
        ax_theta.axhline(target, color="tab:orange", lw=1.4, ls="--",
                         label=f"target {target:.3f} rad")
    if np.isfinite(best_v):
        ax_theta.axvline(best_v, color="tab:green", lw=1.4, ls="--")
        interpolated = int(plot_data.attrs.get("best_is_interpolated", 0))
        how = "interpolated" if interpolated else "nearest measured"
        ax_theta.annotate(
            f"{best_v:.4g} V\n({how})",
            xy=(best_v, ax_theta.get_ylim()[0]), xytext=(4, 6),
            textcoords="offset points", fontsize=9, color="tab:green",
        )

    ax_theta.set_xlabel("coupler flux (V)")
    ax_theta.set_ylabel(r"fitted swap angle $\theta$ (rad)")
    ax_theta.set_title(r"angle calibration $\theta(\Phi_c)$")
    # Only when something was actually labelled: an all-NaN run draws no
    # series, and matplotlib warns on an empty legend.
    if ax_theta.get_legend_handles_labels()[0]:
        ax_theta.legend(loc="best", fontsize=9)

    n_ok = int(plot_data.attrs.get("n_theta_ok", int(ok.sum())))
    fig.suptitle(
        "partial-swap angle calibration     "
        f"({n_ok} of {knob.size} knob values fitted;  "
        r"period in N is $\pi/\theta$)",
        fontsize=12,
    )
    return fig


def _get(plot_data: xr.Dataset, name: str, size: int):
    """A plot_data column, NaN-filled when the variable is absent.

    Keeps this plotter drawable against a plot_data written by an older run that
    predates a variable, rather than raising and taking the raw map down with it.
    """
    if name in plot_data:
        return plot_data[name].values
    return np.full(size, np.nan)
