"""
Readout-fidelity sweep plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ReadoutFidelityEstimator.build_plot_data`` and draws without any recalculation,
so the figures reproduce from the saved ``*_plotdata.nc`` alone. The swept axis
name is taken from ``plot_data.attrs['sweep_coord']`` (e.g. ``amp_prefactor`` or
``frequency``).

Which variables are present depends on the analysis METHOD (``attrs['method']``):
``separation`` and ``mean`` come from every method, everything else is the GMM
method's. Each plotter therefore gates on what it needs, and the estimator only
asks for the figures whose variables exist.

plot_data layout
----------------
coords : <sweep_coord>, ``center``, ``iq``, ``prepared_state``, ``gauss``, ``count``
vars   : ``separation`` (sweep), ``mean`` (sweep, center, iq) — always;
         ``std`` (sweep), ``fidelity`` (sweep), ``snr`` (sweep),
         ``p_outlier``/``norm_res`` (sweep, prepared_state),
         ``gaussian_norms`` (sweep, prepared_state, gauss),
         ``direct_counts`` (sweep, prepared_state, count) — GMM only
attrs  : ``sweep_coord``, ``method``, ``metric``, and (when a best point was
         found) ``best_sweep_value`` / ``best_metric`` / ``best_fidelity``;
         with a companion scale, also ``twin`` (sweep) as a variable plus
         ``twin_label`` / ``best_twin_value``
"""

import matplotlib.pyplot as plt
import numpy as np

from scqat.estimators._twin_axis import add_twin_axis


def _sweep(plot_data):
    coord = plot_data.attrs['sweep_coord']
    return coord, plot_data.coords[coord].values


def _add_twin(ax, plot_data, sweep):
    """Draw the companion scale as a secondary top axis, when one was supplied.
    A figure must never fail over a decoration, so an absent twin is a no-op."""
    if 'twin' not in plot_data:
        return
    add_twin_axis(ax, sweep, plot_data['twin'].values,
                  str(plot_data.attrs.get('twin_label', '')))


def plot_separation_vs_sweep(plot_data):
    """Centre separation and blob width vs the sweep, on ONE axes.

    Both are distances in the same raw IQ units, so the question the pair
    answers — is the separation outgrowing the width? — is read directly instead
    of eyeballed across two figures. The width is GMM-owned: the ``average``
    method fits nothing, so only the separation curve is drawn. The y axis
    starts at 0 because both are magnitudes and their RATIO is the point."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(sweep, plot_data['separation'].values, 'o-', label='|center₀ − center₁|')
    if 'std' in plot_data:
        ax.plot(sweep, plot_data['std'].values, 's--', label='GMM std σ')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('IQ distance', fontsize=14)
    method = str(plot_data.attrs.get('method', ''))
    ax.set_title('Center separation vs sweep' if method == 'average'
                 else 'Center separation and GMM std vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    _add_twin(ax, plot_data, sweep)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_snr_vs_sweep(plot_data):
    """Readout SNR (|center₁ − center₀| / GMM std) as a function of the sweep."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(sweep, plot_data['snr'].values, 'o-')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('SNR (separation / σ)', fontsize=14)
    ax.set_title('Readout SNR vs sweep')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    _add_twin(ax, plot_data, sweep)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_outlier_vs_sweep(plot_data):
    """Outlier probability per prepared_state vs the sweep."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    p_out = plot_data['p_outlier'].values
    for p in range(p_out.shape[1]):
        ax.plot(sweep, p_out[:, p], 'o-', label=f'prepared_state={p}')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('Outlier probability', fontsize=14)
    ax.set_title('Outlier probability vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_norm_res_vs_sweep(plot_data):
    """Normalised fit residue per prepared_state vs the sweep."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    nr = plot_data['norm_res'].values
    for p in range(nr.shape[1]):
        ax.plot(sweep, nr[:, p], 'o-', label=f'prepared_state={p}')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('res / density', fontsize=14)
    ax.set_title('Normalised residue vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_fidelity_vs_sweep(plot_data):
    """Correct-assignment fidelity vs the sweep: the mean of the ``direct_counts``
    diagonal (the reduced ``fidelity`` curve, bold), with the per-state diagonals of
    ``direct_counts`` and ``gaussian_norms`` overlaid. A vertical marker shows the
    chosen ``best_sweep_value`` (from attrs) when present."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    if 'fidelity' in plot_data:
        ax.plot(sweep, plot_data['fidelity'].values, 'k-o', lw=2, label='mean fidelity')

    if 'direct_counts' in plot_data:
        dc = plot_data['direct_counts'].values  # (sweep, prepared_state, count)
        n = min(dc.shape[1], dc.shape[2])
        for k in range(n):
            ax.plot(sweep, dc[:, k, k], 'o-', alpha=0.7, label=f'direct counts state {k}')

    if 'gaussian_norms' in plot_data:
        gn = plot_data['gaussian_norms'].values  # (sweep, prepared_state, gauss)
        n = min(gn.shape[1], gn.shape[2])
        for k in range(n):
            ax.plot(sweep, gn[:, k, k], '--', alpha=0.7, label=f'gaussian norm state {k}')

    best = plot_data.attrs.get('best_sweep_value')
    if best is not None:
        # state the answer in both frames when a companion scale is present
        twin_best = plot_data.attrs.get('best_twin_value')
        label = f'best {coord}={best:.4g}'
        if twin_best is not None:  # the twin label already names the scale on the axis
            label += f' (abs {twin_best:.4g})'
        ax.axvline(best, color='red', ls=':', lw=1.5, label=label)

    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('Fidelity (correct assignment)', fontsize=14)
    ax.set_title('Readout fidelity vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _add_twin(ax, plot_data, sweep)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_means_on_iq_plane(plot_data):
    """Trained GMM centers in the I/Q plane, coloured by the sweep value."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    mean = plot_data['mean'].values  # (sweep, center, iq)
    sc = None
    for c in range(mean.shape[1]):
        sc = ax.scatter(mean[:, c, 0], mean[:, c, 1], c=sweep, cmap='viridis',
                        s=40, marker='o' if c == 0 else '^', label=f'center {c}')
    if sc is not None:
        fig.colorbar(sc, ax=ax, label=coord)
    ax.set_xlabel('I', fontsize=14)
    ax.set_ylabel('Q', fontsize=14)
    ax.set_title('GMM centers on IQ plane vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_response_vs_frequency(plot_data):
    """Resonator transmission magnitude |IQ| for states |g> and |e> vs frequency.

    Displays the response curve dips corresponding to f_dress0 (|g>) and f_dress1 (|e>),
    along with vertical markers for the dips and the optimal readout frequency."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    if 'mean' in plot_data:
        mean = plot_data['mean'].values  # (sweep, center, iq)
        if mean.ndim == 3 and mean.shape[1] >= 2:
            iq_abs_g = np.linalg.norm(mean[:, 0, :], axis=-1)
            iq_abs_e = np.linalg.norm(mean[:, 1, :], axis=-1)
            ax.plot(sweep, iq_abs_g, 'o-', color='tab:blue', label='|g> (|0>)')
            ax.plot(sweep, iq_abs_e, 's-', color='tab:orange', label='|e> (|1>)')

    dip0 = plot_data.attrs.get('detuning_dress0')
    if dip0 is not None and np.isfinite(dip0):
        ax.axvline(dip0, color='tab:blue', ls='--', lw=1.5,
                   label=f'g dip: {dip0:.4g}')

    dip1 = plot_data.attrs.get('detuning_dress1')
    if dip1 is not None and np.isfinite(dip1):
        ax.axvline(dip1, color='tab:orange', ls='--', lw=1.5,
                   label=f'e dip: {dip1:.4g}')

    best = plot_data.attrs.get('best_sweep_value')
    if best is not None and np.isfinite(best):
        ax.axvline(best, color='red', ls=':', lw=1.5,
                   label=f'opt readout: {best:.4g}')

    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('|IQ| magnitude', fontsize=14)
    ax.set_title('Resonator response |IQ| vs frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _add_twin(ax, plot_data, sweep)
    fig.tight_layout()
    plt.close(fig)
    return fig

