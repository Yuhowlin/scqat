"""Phasor Ramsey plotting helpers.

Both functions consume the **plot_data** Dataset built by
``RamseyPhasorEstimator.build_plot_data`` and draw without any recalculation.

plot_data layout
----------------
coords : ``idle_time`` (s), ``frame`` (turns)
vars   : ``signal`` (idle_time, frame); ``contrast`` / ``best_fit`` / ``phase`` /
         ``phase_wrapped`` / ``phase_best_fit`` (idle_time)
attrs  : ``success``, ``railed``, ``t2_star_s``, ``t2_star_err_s``,
         ``stretch_p``, ``stretch_p_err``, ``amp``, ``offset``,
         ``var_explained``, ``detuning_error_hz``, ``detuning_error_err_hz``,
         ``n_phase_valid``

The idle axis is LOG-spaced by construction, so every panel that sets a log
scale first pins explicit positive limits from the finite positive samples. An
all-NaN series under ``set_xscale("log")`` + ``tight_layout()`` is exactly the
crash that once cost a real run all of its PNGs.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _log_xlim(ax, x) -> None:
    """Pin positive limits before a log scale so an empty/NaN series cannot make
    ``tight_layout`` autoscale an unpopulated axes."""
    ax.set_xscale("log")
    positive = np.asarray(x)[np.isfinite(x) & (np.asarray(x) > 0)]
    if positive.size:
        ax.set_xlim(float(positive.min()), float(positive.max()))


def _coherence_text(attrs: dict) -> str:
    t2 = float(attrs.get("t2_star_s", float("nan"))) * 1e6
    t2_err = float(attrs.get("t2_star_err_s", float("nan"))) * 1e6
    p_val = float(attrs.get("stretch_p", float("nan")))
    p_err = float(attrs.get("stretch_p_err", float("nan")))
    lines = [
        f"success: {bool(attrs.get('success', 0))}",
        f"T2* = {t2:.4g} +/- {t2_err:.2g} us",
        f"p = {p_val:.3g} +/- {p_err:.2g}",
        f"var explained = {float(attrs.get('var_explained', float('nan'))):.3f}",
    ]
    if int(attrs.get("railed", 0)):
        lines.append("(a fit parameter railed)")
    return "\n".join(lines)


def _phase_text(attrs: dict) -> str:
    det = float(attrs.get("detuning_error_hz", float("nan")))
    det_err = float(attrs.get("detuning_error_err_hz", float("nan")))
    return "\n".join([
        f"detuning error = {det * 1e-3:.4g} +/- {det_err * 1e-3:.2g} kHz",
        f"points unwrapped = {int(attrs.get('n_phase_valid', 0))}",
    ])


def plot_coherence(plot_data: xr.Dataset) -> plt.Figure:
    """The lock-in contrast (the coherence envelope) against idle time on a log
    axis, with the stretched-exponential overlay and the fitted T2* annotated."""
    t_us = plot_data.coords["idle_time"].values * 1e6
    contrast = plot_data["contrast"].values
    best_fit = plot_data["best_fit"].values

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=100)

    # the raw envelope draws unconditionally; only the overlay is fit-dependent
    ax.plot(t_us, contrast, ".", markersize=5, label="lock-in contrast |z|")
    if np.isfinite(best_fit).any():
        ax.plot(t_us, best_fit, "-", linewidth=2, label="stretched exp fit")
    ax.set_xlabel("Idle time (us)", fontsize=14)
    ax.set_ylabel("Contrast |z|", fontsize=14)
    ax.set_title("phasor Ramsey coherence")
    _log_xlim(ax, t_us)
    ax.legend()

    ax.text(
        0.02, 0.02, _coherence_text(plot_data.attrs),
        transform=ax.transAxes, fontsize=11,
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_phase(plot_data: xr.Dataset) -> plt.Figure:
    """Diagnostic: the raw fringe map over (idle_time, frame) beside the lock-in
    phase, with the trusted unwrapped prefix and its fitted slope."""
    t_us = plot_data.coords["idle_time"].values * 1e6
    frame = plot_data.coords["frame"].values
    signal = plot_data["signal"].values
    wrapped = plot_data["phase_wrapped"].values
    phase = plot_data["phase"].values
    phase_fit = plot_data["phase_best_fit"].values

    fig, (ax_map, ax_phase) = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

    # raw fringe map -- always drawn, never fit-dependent
    ax_map.pcolormesh(t_us, frame, np.asarray(signal, dtype=float).T, shading="auto")
    ax_map.set_xlabel("Idle time (us)", fontsize=14)
    ax_map.set_ylabel("Frame (turns)", fontsize=14)
    ax_map.set_title("raw fringe map")
    _log_xlim(ax_map, t_us)

    ax_phase.plot(t_us, wrapped, ".", markersize=4, color="0.7",
                  label="wrapped angle(z)")
    if np.isfinite(phase).any():
        ax_phase.plot(t_us, phase, ".-", markersize=4, linewidth=1,
                      label="unwrapped (trusted prefix)")
    if np.isfinite(phase_fit).any():
        ax_phase.plot(t_us, phase_fit, "-", linewidth=2, label="slope fit")
    ax_phase.set_xlabel("Idle time (us)", fontsize=14)
    ax_phase.set_ylabel("Phase (rad)", fontsize=14)
    ax_phase.set_title("lock-in phase")
    _log_xlim(ax_phase, t_us)
    ax_phase.legend(fontsize=9)

    ax_phase.text(
        0.02, 0.98, _phase_text(plot_data.attrs),
        transform=ax_phase.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    plt.close(fig)
    return fig
