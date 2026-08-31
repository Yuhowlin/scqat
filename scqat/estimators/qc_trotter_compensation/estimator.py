"""Which AC-Stark compensation amplitude the Trotter chain wants -- record-only.

The chain repeats a Trotter step (partial swap source->relay, partial swap
relay->sink, parametric reset of the relay, per-qubit AC-Stark phase
compensation) ``N`` times. The sink does not simply collect the source: it
accumulates amplitude from EVERY round, so the round-to-round phase decides
whether those contributions add or cancel. Propagating the single-excitation
amplitudes through one round gives

    |K_M| = sin(t1) sin(t2) |S0| * | SUM_j cos^(M-1-j)(t2) cos^j(t1) e^{i j dPhi} |

with ``dPhi = phi_source - phi_sink``. Three things follow, and this estimator is
shaped by all three:

* only the DIFFERENTIAL phase is observable -- a common-mode phase cancels, so
  ONE compensation amplitude is swept, not one per qubit;
* the sum peaks at ``dPhi = 0``, so the sweep has a single optimum;
* when the rounds cancel, only the LAST round survives and the sink peaks at
  ``N = 1``; when they add, the peak moves out to several rounds. So
  ``n_at_max`` reads the phase condition more sharply than the peak height
  does, and it is reported beside it rather than buried.

Record-only: this estimator proposes nothing and fits no model to the sweep. It
locates the optimum, reports both discriminators, and draws the map -- the
SUCCESS verdict stays in SCQO.

Dataset contract:
  vars   : ``population`` -- dims ``(qubit, compensation_amp, round_count)`` in
           any order: the averaged marginal P(level >= 1) of each chain qubit.
  coords : ``qubit`` (chain qubit names, chain order) / ``compensation_amp``
           (dimensionless factor of the stark operation's baked amplitude) /
           ``round_count`` (dimensionless Trotter-step count N)
  kwargs : ``source`` / ``relay`` / ``sink`` -- chain role names; ``sink`` picks
           the trace the optimum is read from and is what makes the answer
           meaningful. ``compensation_target`` names the qubit whose tone was
           swept, for the figure only.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.qc_trotter_compensation.visualization import (
    plot_compensation_scan,
    plot_sink_map,
)

QUBIT_DIM = "qubit"
AMP_AXIS = "compensation_amp"
ROUND_AXIS = "round_count"

#: chain role kwargs -- the labels the figure reads back.
ROLES = ("source", "relay", "sink")

#: figure keys. ``save_figures`` prefixes the estimator name unless the key IS
#: it, so these land as ``qc_trotter_compensation.png`` and
#: ``qc_trotter_compensation_map.png``.
FIG_SCAN = "qc_trotter_compensation"
FIG_MAP = "map"


def _roles(kwargs: Dict[str, Any]) -> Dict[str, str]:
    """The chain role names supplied by the caller, blank when unknown."""
    return {role: str(kwargs.get(role) or "") for role in ROLES}


def _qubit_names(dataset: xr.Dataset) -> List[str]:
    return [str(v) for v in np.atleast_1d(dataset[QUBIT_DIM].values)]


def _peak_per_amp(rounds: np.ndarray, sink: np.ndarray) -> Dict[str, np.ndarray]:
    """Per compensation amplitude: the sink peak and the N at which it occurs.

    ``sink`` is ``(amp, round)``. An all-NaN row (a failed acquisition) degrades
    to NaN rather than raising, so the map still draws.
    """
    n_amp = sink.shape[0]
    p_max = np.full(n_amp, np.nan)
    n_at_max = np.full(n_amp, np.nan)
    p_final = np.full(n_amp, np.nan)
    for i in range(n_amp):
        row = sink[i]
        if not np.isfinite(row).any():
            continue
        j = int(np.nanargmax(row))
        p_max[i] = float(row[j])
        n_at_max[i] = float(rounds[j])
        p_final[i] = float(row[-1])
    return {"sink_p_max": p_max, "sink_n_at_max": n_at_max, "sink_p_final": p_final}


class QcTrotterCompensationEstimator(BaseEstimator):
    """Locate the compensation amplitude that maximises chain transport."""

    estimator_name = "qc_trotter_compensation"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "population" not in dataset.data_vars:
            raise ValueError(
                "qc_trotter_compensation estimator requires the population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in (QUBIT_DIM, AMP_AXIS, ROUND_AXIS):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_trotter_compensation estimator requires a {axis!r} coordinate"
                )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        roles = _roles(kwargs)
        qubits = _qubit_names(dataset)
        amps = np.asarray(dataset[AMP_AXIS].values, dtype=float)
        rounds = np.asarray(dataset[ROUND_AXIS].values, dtype=float)
        pop = np.asarray(
            dataset["population"]
            .transpose(QUBIT_DIM, AMP_AXIS, ROUND_AXIS).values,
            dtype=float,
        )

        results: Dict[str, Any] = {
            "qubits": qubits,
            "n_compensation_amp": int(amps.size),
            "n_round_count": int(rounds.size),
            "compensation_target": str(kwargs.get("compensation_target") or ""),
            **roles,
        }

        sink = roles["sink"]
        if sink not in qubits:
            # Without the sink there is no transport to optimise. Say so in the
            # results rather than guessing a trace -- SCQO turns this into FAILED.
            results.update({
                "success": False,
                "best_compensation_amp": float("nan"),
                "best_sink_p_max": float("nan"),
                "best_n_at_max": float("nan"),
                "worst_compensation_amp": float("nan"),
                "worst_sink_p_max": float("nan"),
                "contrast": float("nan"),
            })
            results["_sink"] = np.full((amps.size, rounds.size), np.nan)
            results["_pop"] = pop
            results["_curves"] = {k: v for k, v in _peak_per_amp(
                rounds, np.full((amps.size, rounds.size), np.nan)).items()}
            return results

        sink_map = pop[qubits.index(sink)]                      # (amp, round)
        curves = _peak_per_amp(rounds, sink_map)
        p_max = curves["sink_p_max"]

        if np.isfinite(p_max).any():
            best = int(np.nanargmax(p_max))
            worst = int(np.nanargmin(p_max))
            results.update({
                "success": True,
                "best_compensation_amp": float(amps[best]),
                "best_sink_p_max": float(p_max[best]),
                "best_n_at_max": float(curves["sink_n_at_max"][best]),
                "best_sink_p_final": float(curves["sink_p_final"][best]),
                "worst_compensation_amp": float(amps[worst]),
                "worst_sink_p_max": float(p_max[worst]),
                # How much the phase knob is worth on this chain: 1.0 means the
                # sweep found no dependence at all, so compensation buys nothing.
                "contrast": float(p_max[best] / p_max[worst])
                if p_max[worst] > 0 else float("nan"),
            })
            # Every qubit's trace AT the optimum -- the population-vs-N curve the
            # run exists to produce, so it lands in the metadata, not only a figure.
            results["per_qubit_at_best"] = {
                q: [float(v) for v in pop[k, best]] for k, q in enumerate(qubits)
            }
        else:
            results.update({
                "success": False,
                "best_compensation_amp": float("nan"),
                "best_sink_p_max": float("nan"),
                "best_n_at_max": float("nan"),
                "worst_compensation_amp": float("nan"),
                "worst_sink_p_max": float("nan"),
                "contrast": float("nan"),
            })

        results["compensation_amp"] = [float(v) for v in amps]
        results["round_count"] = [float(v) for v in rounds]
        for key, values in curves.items():
            results[key] = [float(v) for v in values]
        # underscore-prefixed: bulky arrays kept for build_plot_data only
        results["_sink"] = sink_map
        results["_pop"] = pop
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the bulky maps; the per-amp curves and the optimum stay."""
        return {k: v for k, v in results.items() if not k.startswith("_")}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """The measured arrays, carried unconditionally (there is no fit to fail)."""
        roles = _roles(kwargs)
        qubits = _qubit_names(dataset)
        amps = np.asarray(dataset[AMP_AXIS].values, dtype=float)
        rounds = np.asarray(dataset[ROUND_AXIS].values, dtype=float)
        pop = results.get("_pop")
        if pop is None:
            pop = np.asarray(
                dataset["population"]
                .transpose(QUBIT_DIM, AMP_AXIS, ROUND_AXIS).values,
                dtype=float,
            )
        sink_map = results.get("_sink")
        if sink_map is None:
            sink = roles["sink"]
            sink_map = (pop[qubits.index(sink)] if sink in qubits
                        else np.full((amps.size, rounds.size), np.nan))

        n_amp = amps.size
        out = xr.Dataset(
            {
                "population": ((QUBIT_DIM, AMP_AXIS, ROUND_AXIS),
                               np.asarray(pop, dtype=float)),
                "sink": ((AMP_AXIS, ROUND_AXIS), np.asarray(sink_map, dtype=float)),
                "sink_p_max": ((AMP_AXIS,),
                               _as_row(results.get("sink_p_max"), n_amp)),
                "sink_n_at_max": ((AMP_AXIS,),
                                  _as_row(results.get("sink_n_at_max"), n_amp)),
                "sink_p_final": ((AMP_AXIS,),
                                 _as_row(results.get("sink_p_final"), n_amp)),
            },
            coords={QUBIT_DIM: qubits, AMP_AXIS: amps, ROUND_AXIS: rounds},
        )
        # netCDF-safe attrs only: bools as int, absent roles as empty strings.
        out.attrs.update({
            "success": int(bool(results.get("success", False))),
            "best_compensation_amp": float(
                results.get("best_compensation_amp", float("nan"))),
            "best_sink_p_max": float(results.get("best_sink_p_max", float("nan"))),
            "best_n_at_max": float(results.get("best_n_at_max", float("nan"))),
            "worst_compensation_amp": float(
                results.get("worst_compensation_amp", float("nan"))),
            "contrast": float(results.get("contrast", float("nan"))),
            "compensation_target": str(results.get("compensation_target", "")),
            **roles,
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
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        return render_figures(
            {
                FIG_SCAN: lambda: plot_compensation_scan(plot_data),
                FIG_MAP: lambda: plot_sink_map(plot_data),
            },
            label=self.estimator_name,
        )


def _as_row(values, size: int) -> np.ndarray:
    """A per-amplitude column of the right length, NaN-filled when absent.

    ``build_plot_data`` may be called standalone (the replot path) with a results
    dict that never held the curves, so a missing one degrades rather than
    raising -- rule 1 of "raw data must always be plottable".
    """
    if values is None:
        return np.full(size, np.nan)
    return np.asarray(values, dtype=float)
