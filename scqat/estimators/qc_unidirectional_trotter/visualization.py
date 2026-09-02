"""Figures for the unidirectional-coupling Trotter chain.

Both plotters draw from ``plot_data`` ONLY (the estimator-output-contract rule)
and both are pure RAW views — there is no fit to overlay, so neither may assume
a finite value anywhere: the population axis is pinned to 0..1 rather than taken
from the data, and an all-NaN acquisition renders an annotated empty panel
instead of raising (which would cost the run every other figure).
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["plot_chain_populations", "plot_chain_joint"]

#: chain roles, in chain order — the words appended to a qubit's legend entry.
_ROLE_TAGS = ("source", "relay", "sink")


def _role_of(plot_data: xr.Dataset) -> Dict[str, str]:
    """``{qubit name: role word}`` from the plot-data attrs (may be empty)."""
    out: Dict[str, str] = {}
    for role in _ROLE_TAGS:
        name = str(plot_data.attrs.get(role, "") or "")
        if name:
            out[name] = role
    return out


def _names(plot_data: xr.Dataset, dim: str) -> List[str]:
    return [str(v) for v in np.atleast_1d(plot_data[dim].values)]


def _annotate_empty(ax: plt.Axes, message: str) -> None:
    """A panel with nothing finite to draw still has to render."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def plot_chain_populations(plot_data: xr.Dataset) -> plt.Figure:
    """P(|1>) against the Trotter-step count N, one trace per chain qubit.

    The transport picture: the source decays, the sink grows, and the relay
    stays small because it is reset every round."""
    rounds = np.asarray(plot_data["round_count"].values, dtype=float)
    qubits = _names(plot_data, "qubit")
    pop = np.asarray(
        plot_data["population"].transpose("qubit", "round_count").values, dtype=float
    )
    role_of = _role_of(plot_data)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    drawn = False
    for k, qubit in enumerate(qubits):
        role = role_of.get(qubit, "")
        label = f"{qubit} ({role})" if role else qubit
        ax.plot(rounds, pop[k], marker="o", markersize=4, linewidth=1.5, label=label)
        drawn = drawn or bool(np.isfinite(pop[k]).any())
    ax.set_xlabel("Trotter steps N")
    ax.set_ylabel("P(|1>)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend()
    else:
        _annotate_empty(ax, "no finite population data")
    ax.set_title("unidirectional Trotter chain — excitation transport vs N")
    return fig


def plot_chain_joint(plot_data: xr.Dataset) -> plt.Figure:
    """The joint chain distribution against N — one row per basis state.

    Present only when the run kept every shot; each label's digit order is the
    chain-qubit order (leftmost digit = the first ``qubit``).

    ``imshow`` rather than ``pcolormesh``: the round axis is an integer grid, so
    the extent is exact, and a single-column map (``max_rounds=0``) still draws
    instead of failing on cell-boundary inference."""
    rounds = np.asarray(plot_data["round_count"].values, dtype=float)
    labels = _names(plot_data, "joint_state")
    joint = np.asarray(
        plot_data["joint_population"].transpose("joint_state", "round_count").values,
        dtype=float,
    )
    qubits = _names(plot_data, "qubit")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if rounds.size == 0 or not labels:
        _annotate_empty(ax, "no joint-population data")
        return fig
    im = ax.imshow(
        joint,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(rounds[0] - 0.5, rounds[-1] + 0.5, -0.5, len(labels) - 0.5),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Trotter steps N")
    ket = "".join(qubits)
    ax.set_ylabel(f"joint state  |{ket}>" if ket else "joint state")
    fig.colorbar(im, ax=ax, label="joint population")
    if not np.isfinite(joint).any():
        _annotate_empty(ax, "no finite joint-population data")
    ax.set_title("unidirectional Trotter chain — joint populations vs N")
    return fig
