"""Excitation transport along a Trotterized unidirectional chain — record-only.

The sequence excites ONE qubit of a three-qubit chain, then repeats a Trotter
step (partial swap source->relay, partial swap relay->sink, parametric reset of
the relay, per-qubit AC-Stark phase compensation) ``N`` times before reading
every qubit out in the same shot. This estimator does NOT fit the transport: it
draws the measured populations against ``N`` and extracts a small
self-describing summary of where each qubit's excitation peaks. The SUCCESS /
``min_transfer`` verdict stays in SCQO.

The relay reset is what makes the coupling one-way, so the picture to read off
the figure is: the source decays, the sink grows, and the relay stays small.

Unlike the pair estimators (``qc_n_swap_amp``, ``pair_swap_chevron``) this one
consumes the WHOLE multi-qubit dataset rather than a per-target slice: the joint
panel is a cross-qubit quantity, and a per-target split cannot draw it.

Dataset contract:
  vars   : ``population`` — dims ``(qubit, round_count)`` in any order: the
           averaged marginal P(level >= 1) of each chain qubit.
           ``joint_population`` — OPTIONAL, dims ``(joint_state, round_count)``:
           the joint distribution over the chain's basis states, present only
           when the run kept every shot (SCQO's ``readout_mode="shot"``).
  coords : ``qubit`` (chain qubit names, chain order) / ``round_count``
           (dimensionless Trotter-step count N) / ``joint_state`` (per-qubit
           level digits, leftmost digit = the first ``qubit``) when joint.
  kwargs : ``source`` / ``relay`` / ``sink`` — chain role names used to label
           the figure and to pick the transport summary; all optional.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.qc_unidirectional_trotter.visualization import (
    plot_chain_joint,
    plot_chain_populations,
)

QUBIT_DIM = "qubit"
AXIS = "round_count"
JOINT_DIM = "joint_state"

#: chain role kwargs, in chain order — the labels the figure reads back.
ROLES = ("source", "relay", "sink")

#: figure keys — stable across runs and readout modes, so a saved plotdata.nc
#: always replots under the same PNG names. ``save_figures`` prefixes the
#: estimator name unless the key IS it, so these land as
#: ``qc_unidirectional_trotter.png`` and ``qc_unidirectional_trotter_joint.png``.
FIG_POPULATIONS = "qc_unidirectional_trotter"
FIG_JOINT = "joint"


def _roles(kwargs: Dict[str, Any]) -> Dict[str, str]:
    """The chain role names supplied by the caller, blank when unknown."""
    return {role: str(kwargs.get(role) or "") for role in ROLES}


def _qubit_names(dataset: xr.Dataset) -> List[str]:
    return [str(v) for v in np.atleast_1d(dataset[QUBIT_DIM].values)]


def _trace_summary(rounds: np.ndarray, trace: np.ndarray) -> Dict[str, float]:
    """One qubit's population-vs-N summary. All-NaN degrades to NaN, never raises."""
    trace = np.asarray(trace, dtype=float)
    out = {
        "p_initial": float(trace[0]) if trace.size else float("nan"),
        "p_final": float(trace[-1]) if trace.size else float("nan"),
    }
    if not np.isfinite(trace).any():
        out.update(p_max=float("nan"), p_min=float("nan"), n_at_max=float("nan"))
        return out
    i = int(np.nanargmax(trace))
    out.update(
        p_max=float(trace[i]),
        p_min=float(np.nanmin(trace)),
        n_at_max=float(rounds[i]),
    )
    return out


class QcUnidirectionalTrotterEstimator(BaseEstimator):
    """Draw the chain's transport curves (and joint distribution) vs N."""

    estimator_name = "qc_unidirectional_trotter"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "population" not in dataset.data_vars:
            raise ValueError(
                "qc_unidirectional_trotter estimator requires the population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in (QUBIT_DIM, AXIS):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_unidirectional_trotter estimator requires a {axis!r} coordinate"
                )
        if "joint_population" in dataset.data_vars and JOINT_DIM not in dataset.coords:
            raise ValueError(
                "qc_unidirectional_trotter estimator requires a "
                f"{JOINT_DIM!r} coordinate alongside joint_population"
            )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        roles = _roles(kwargs)
        qubits = _qubit_names(dataset)
        rounds = np.asarray(dataset[AXIS].values, dtype=float)
        pop = np.asarray(
            dataset["population"].transpose(QUBIT_DIM, AXIS).values, dtype=float
        )

        per_qubit = {q: _trace_summary(rounds, pop[k]) for k, q in enumerate(qubits)}
        results: Dict[str, Any] = {
            "qubits": qubits,
            "n_round_count": int(rounds.size),
            "max_round_count": float(rounds[-1]) if rounds.size else float("nan"),
            "per_qubit": per_qubit,
            **roles,
        }
        # The transport headline, named rather than left to a reader of
        # per_qubit: SCQO's min_transfer verdict is made against exactly this.
        sink = roles["sink"]
        if sink in per_qubit:
            results["sink_p_max"] = per_qubit[sink]["p_max"]
            results["sink_n_at_max"] = per_qubit[sink]["n_at_max"]
            results["sink_p_final"] = per_qubit[sink]["p_final"]

        has_joint = "joint_population" in dataset.data_vars
        results["has_joint"] = bool(has_joint)
        if has_joint:
            joint = dataset["joint_population"].transpose(JOINT_DIM, AXIS)
            labels = [str(v) for v in np.atleast_1d(joint[JOINT_DIM].values)]
            values = np.asarray(joint.values, dtype=float)
            results["joint_p_max"] = {
                label: (float(np.nanmax(row)) if np.isfinite(row).any() else float("nan"))
                for label, row in zip(labels, values)
            }
        return results

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """The measured arrays, carried unconditionally (there is no fit to fail)."""
        roles = _roles(kwargs)
        qubits = _qubit_names(dataset)
        rounds = np.asarray(dataset[AXIS].values, dtype=float)
        pop = np.asarray(
            dataset["population"].transpose(QUBIT_DIM, AXIS).values, dtype=float
        )
        out = xr.Dataset(
            {"population": ((QUBIT_DIM, AXIS), pop)},
            coords={QUBIT_DIM: qubits, AXIS: rounds},
        )
        has_joint = "joint_population" in dataset.data_vars
        if has_joint:
            joint = dataset["joint_population"].transpose(JOINT_DIM, AXIS)
            out["joint_population"] = (
                (JOINT_DIM, AXIS),
                np.asarray(joint.values, dtype=float),
            )
            out = out.assign_coords(
                {JOINT_DIM: [str(v) for v in np.atleast_1d(joint[JOINT_DIM].values)]}
            )
        # netCDF-safe attrs only: the bool as int, absent roles as empty strings.
        out.attrs.update({"has_joint": int(has_joint), **roles})
        return out

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        builders = {FIG_POPULATIONS: lambda: plot_chain_populations(plot_data)}
        if "joint_population" in plot_data.data_vars:
            builders[FIG_JOINT] = lambda: plot_chain_joint(plot_data)
        return render_figures(builders, label=self.estimator_name)
