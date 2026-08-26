"""
Parametric-drive decoherence plotting helpers.

All three functions consume the **plot_data** Dataset built by
``ParametricDriveDecoherenceEstimator.build_plot_data`` and draw without any
recalculation.

* :func:`plot_rho11_map` — the RAW 2-D map, driving frequency x driving time,
  coloured by population. Drawn from ``rho11_data``, which is filled at every
  frequency whether or not that frequency's fit converged, so this figure is
  the one that always has something to show.
* :func:`plot_rho11_fits` — the same rho_11 as per-frequency traces, with the
  fitted curve overlaid where it exists.
* :func:`plot_decoherence_params` — the pure-FIT view: gamma, lambda, |Delta|
  and the EP figure of merit vs driving frequency.

plot_data layout
----------------
coords : ``driving_frequency`` (Hz), ``driving_time`` (ns)
vars   : per-frequency scalars ``gamma`` / ``gamma_err`` / ``lambda_`` /
         ``lambda_err`` / ``Delta`` / ``Delta_err`` / ``rho_0`` / ``rho_0_err`` /
         ``chisqr`` / ``ep_metric`` / ``success``; 2-D maps ``rho11_data`` /
         ``rho11_fit`` (driving_frequency, driving_time)
attrs  : ``has_tomography``, ``n_freq``, ``n_decoh_ok``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#: how far outside [0, 1] a value may sit and still count as a population. The
#: estimator accepts a raw quadrature as a last resort (its state-variable
#: candidates end in ``I``), and volts are not a population — the map falls back
#: to autoscaling rather than rendering a blank panel under a 0-1 clamp.
_POPULATION_TOL = 0.05


def _value_ylim(y, pad: float = 0.08):
    """Axis limits from the VALUES alone — ``None`` when there are none.

    ``plot_decoherence_params`` draws error bars, and matplotlib's autoscale
    counts the bar CAPS as data. One non-converged frequency can carry a
    ``gamma_err`` orders of magnitude past the value range, which rescales the
    panel and flattens every real point into a flat line. Limits therefore come
    from the finite values only; the bars are still drawn and simply clip at the
    axes, so a bad fit still reads as a bar running off-panel.

    The scalar arrays are pre-filled ``np.full(n_freq, np.nan)`` and only
    overwritten where the fit converged, so an all-failed run reaches here
    all-NaN — hence the empty check rather than a bare ``np.nanmin``, which
    would warn and return NaN limits."""
    finite = np.asarray(y, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:  # a single converged frequency — do not ask for a zero-height axis
        span = abs(lo) or 1.0
        return lo - 0.5 * span, hi + 0.5 * span
    span = hi - lo
    return lo - pad * span, hi + pad * span


def plot_rho11_map(plot_data: xr.Dataset) -> plt.Figure:
    """The RAW 2-D map: driving frequency (x) x driving time (y), coloured by
    population.

    This is the figure that survives everything — ``rho11_data`` is written for
    every frequency before the per-frequency fit is even attempted, so a run
    where every fit failed still draws its chevron here.

    ``rho11_data`` is stored ``(driving_frequency, driving_time)`` and
    ``pcolormesh`` wants ``(len(y), len(x))``, hence the transpose."""
    f_mhz = plot_data.coords["driving_frequency"].values.astype(float) / 1e6
    t_ns = plot_data.coords["driving_time"].values.astype(float)
    rho = plot_data["rho11_data"].values  # (driving_frequency, driving_time)

    # A population is bounded, so fix the scale at [0, 1] and make runs
    # comparable; a raw quadrature is not, and clamping volts to [0, 1] would
    # render a blank panel for a run that is already wrong.
    finite = rho[np.isfinite(rho)]
    is_population = bool(
        finite.size
        and finite.min() >= -_POPULATION_TOL
        and finite.max() <= 1.0 + _POPULATION_TOL
    )
    limits = {"vmin": 0.0, "vmax": 1.0} if is_population else {}

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    pcm = ax.pcolormesh(f_mhz, t_ns, rho.T, shading="auto", cmap="viridis", **limits)
    fig.colorbar(pcm, ax=ax,
                 label=r"$\rho_{11}$" if is_population else "Signal (arb. u.)")
    ax.set_xlabel("Driving frequency (MHz)")
    ax.set_ylabel("Driving time (ns)")

    kind = "tomography" if plot_data.attrs.get("has_tomography", 0) else r"$\rho_{11}$-only"
    note = "" if is_population else " — NOT a population (raw quadrature?)"
    ax.set_title(f"Parametric-drive chevron [{kind}]{note}")
    fig.tight_layout()
    return fig


def plot_decoherence_params(plot_data: xr.Dataset) -> plt.Figure:
    """4-panel summary of the fitted decoherence parameters vs driving frequency:
    γ, λ, |Δ| (with error bars) and the EP figure of merit 8λ²/γ².

    The three error-bar panels scale to their VALUES, not their bars — see
    :func:`_value_ylim`. The EP panel is deliberately left on autoscale: it
    carries no error bars, and a large 8λ²/γ² is exactly what it exists to
    show."""
    f_mhz = plot_data.coords["driving_frequency"].values.astype(float) / 1e6
    gamma = plot_data["gamma"].values
    lam = plot_data["lambda_"].values
    delta = np.abs(plot_data["Delta"].values)
    ep = plot_data["ep_metric"].values

    g_err = plot_data["gamma_err"].values
    l_err = plot_data["lambda_err"].values
    d_err = plot_data["Delta_err"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=120)
    panels = [
        (axes[0, 0], gamma, g_err, r"$\gamma$ (1/ns)", r"relaxation rate $\gamma$"),
        (axes[0, 1], lam, l_err, r"$\lambda$ (1/ns)", r"coupling $\lambda$"),
        (axes[1, 0], delta, d_err, r"$|\Delta|$ (1/ns)", r"detuning $|\Delta|$"),
    ]
    for ax, y, yerr, ylabel, title in panels:
        ax.errorbar(f_mhz, y, yerr=yerr, fmt="o-", ms=4, capsize=2)
        # the ERROR BARS must not set the scale (see _value_ylim)
        ylim = _value_ylim(y)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Driving frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(f_mhz, ep, "s-", ms=4, color="C3")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="EP ($8\\lambda^2/\\gamma^2=1$)")
    ax.set_xlabel("Driving frequency (MHz)")
    ax.set_ylabel(r"$8\lambda^2/\gamma^2$")
    ax.set_title("exceptional-point figure of merit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    n_ok = int(plot_data.attrs.get("n_decoh_ok", int(np.isfinite(gamma).sum())))
    n_freq = int(plot_data.attrs.get("n_freq", f_mhz.size))
    kind = "tomography" if plot_data.attrs.get("has_tomography", 0) else r"$\rho_{11}$-only"
    fig.suptitle(f"Parametric-drive decoherence [{kind}] — fitted {n_ok}/{n_freq} frequencies")
    fig.tight_layout()
    return fig


def plot_rho11_fits(plot_data: xr.Dataset) -> plt.Figure:
    """ρ₁₁(t) data (dots) and decoherence fit (lines), one trace per driving
    frequency, coloured by frequency with a shared colorbar."""
    t = plot_data.coords["driving_time"].values.astype(float)
    freqs = plot_data.coords["driving_frequency"].values.astype(float)
    data = plot_data["rho11_data"].values  # (freq, time)
    fit = plot_data["rho11_fit"].values

    f_mhz = freqs / 1e6
    fmin, fmax = float(f_mhz.min()), float(f_mhz.max())
    norm = plt.Normalize(vmin=fmin, vmax=fmax if fmax > fmin else fmin + 1.0)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for i in range(freqs.size):
        color = cmap(norm(f_mhz[i]))
        ax.plot(t, data[i], "o", ms=2.5, alpha=0.5, color=color)
        if np.isfinite(fit[i]).any():
            ax.plot(t, fit[i], "-", lw=1.2, color=color)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Driving frequency (MHz)")
    ax.set_xlabel("Driving time (ns)")
    ax.set_ylabel(r"$\rho_{11}$")
    ax.set_title(r"$\rho_{11}(t)$ data (dots) and non-Markovian decoherence fit (lines)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
