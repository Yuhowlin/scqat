import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.state_discrimination import state_iq_arrays
from scqat.estimators._twin_axis import TWIN_KNOBS, twin_values
from scqat.estimators.readout_fidelity.methods import METHODS, ReadoutFidelityMethod
from scqat.estimators.readout_fidelity.visualization import (
    plot_outlier_vs_sweep,
    plot_separation_vs_sweep,
    plot_snr_vs_sweep,
    plot_norm_res_vs_sweep,
    plot_fidelity_vs_sweep,
    plot_means_on_iq_plane,
    plot_response_vs_frequency,
)


class ReadoutFidelityEstimator(BaseEstimator):
    """
    Readout-fidelity sweep: run state discrimination at every point of a swept
    readout parameter, summarise how the discrimination quality evolves, and
    report the sweep value that maximises the readout fidelity.

    Expects an xarray.Dataset with:
        - Variables:   ``I``, ``Q``
        - Coordinates: ``shot_idx``, ``prepared_state`` (required by the inner
          :class:`StateDiscriminationEstimator`)
        - Coordinate:  the swept axis named by :attr:`sweep_coord`

    For each value of ``sweep_coord`` the data is sliced and summarised by the
    selected METHOD (``method="gmm"``, the default, or ``"average"`` — see
    :mod:`scqat.estimators.readout_fidelity.methods`), and the summaries are
    collected as a function of the sweep.

    * ``gmm`` hands every slice to the family-shared reduction
      :func:`scqat.tools.discriminate.discriminate_states`, collecting the
      trained std/means, outlier probability, normalised residue, Gaussian norms
      and direct counts. The **fidelity** at each point is the mean of the
      confusion-matrix diagonal (``direct_counts[k, k]`` = fraction of
      prepared-state-k shots assigned to label k), and the answer is the point
      that maximises it.
    * ``average`` fits nothing: each prepared state's centre is the average of
      its I/Q, which also makes it the method for FPGA-averaged data with no
      shot axis at all. With no fitted width there is no fidelity, so the answer
      is the point of largest centre **separation**.

    Either way ``best_sweep_value`` is the chosen point and ``best_metric`` its
    metric value (``metric`` names which curve that is).

    This unifies qcat's near-duplicate ``ROFidelityPower`` (``amp_prefactor``) and
    ``ROFidelityFreq`` (``frequency``); use the :class:`ReadoutPowerFidelityEstimator`
    / :class:`ReadoutFreqFidelityEstimator` subclasses, or set ``sweep_coord``
    directly. (The power-specific linear mean-drift refit from qcat is not ported
    here — see MIGRATION.md.)
    """

    estimator_name = "readout_fidelity"
    sweep_coord: Optional[str] = None  # subclasses set this; or pass sweep_coord kwarg
    fidelity_floor: float = 0.5  # below this the best point is flagged unsuccessful
    #: same name => same meaning AND unit in EVERY method; orchestration (SCQO)
    #: may rely only on these. Validated right after the per-slice loop.
    COMMON_KEYS = (
        'sweep_coord', 'sweep_values', 'method', 'metric', 'mean', 'separation',
        'failed', 'best_index', 'best_sweep_value', 'best_metric',
        'best_separation', 'success',
    )
    #: optional companion scale for the swept axis — a coordinate over the same
    #: points plus its label, drawn as a secondary axis (see
    #: :mod:`scqat.estimators._twin_axis`). Overridable per call via kwargs.
    twin_coord: Optional[str] = None
    twin_label: Optional[str] = None

    # ------------------------------------------------------------------
    def _resolve_coord(self, kwargs: Dict[str, Any]) -> str:
        coord = kwargs.get('sweep_coord', self.sweep_coord)
        if coord is None:
            raise ValueError(
                "ReadoutFidelityEstimator needs a sweep coordinate: set the "
                "'sweep_coord' class attribute (or use a subclass) or pass "
                "sweep_coord=... ."
            )
        return coord

    @staticmethod
    def _resolve_method(kwargs: Dict[str, Any]) -> ReadoutFidelityMethod:
        name = kwargs.get('method', 'gmm')
        try:
            return METHODS[name]
        except KeyError:
            raise ValueError(
                f"Unknown method {name!r}; valid: {sorted(METHODS)}"
            ) from None

    def _check_data(self, dataset: xr.Dataset) -> None:
        """What EVERY method needs. The per-shot requirement is method-owned and
        checked in :meth:`extract_parameters`, which is where the method is
        known (``analyze`` does not forward kwargs here)."""
        for var in ("I", "Q"):
            if var not in dataset:
                raise ValueError(f"Readout fidelity requires a '{var}' variable.")
        if "prepared_state" not in dataset.coords:
            raise ValueError("Readout fidelity requires a 'prepared_state' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Summarise every sweep point with the selected method and pick the
        metric-optimal sweep value.

        Kwargs — flat and fully owned; unknown names raise:
            method (str): ``"gmm"`` (default) or ``"average"``; see
                :mod:`scqat.estimators.readout_fidelity.methods`.
            sweep_coord (str): Override the swept coordinate name.
            user_std / user_mean / outlier_sigma: Knobs of
                :func:`scqat.tools.discriminate.discriminate_states` — ``gmm``
                only; the average method fits nothing and REFUSES them.
            outliers_threshold (float): Selection constraint consumed by
                :class:`ReadoutPowerFidelityEstimator` (accepted, unused here);
                ``gmm`` only, since it is answerable only from p_outlier.
            twin_coord, twin_label: optional companion scale for the swept axis
                (see :mod:`scqat.estimators._twin_axis`) — a coordinate over the
                same points plus its axis label. Absent/non-finite/non-monotone is
                simply not drawn.

        Returns the sweep axis, the per-sweep arrays of the chosen method, and
        the best point. COMMON to every method:
            sweep_coord (name), sweep_values (S,), method, metric,
            mean (S, center, iq), separation (S,), failed (S,), best_index,
            best_sweep_value, best_metric, best_separation, success.
        ``gmm`` adds: std (S,), p_outlier (S, prepared_state),
            norm_res (S, prepared_state), gaussian_norms (S, prepared_state, gauss),
            direct_counts (S, prepared_state, count), fidelity (S,), snr (S,),
            best_fidelity.
        Plus twin_values / twin_label / best_twin_value when a drawable companion
        scale was supplied.
        """
        coord = self._resolve_coord(kwargs)
        method = self._resolve_method(kwargs)
        # Fail loudly BEFORE any per-slice work — a typo'd knob must never be
        # swallowed by the per-slice try/except, and a knob belonging to the
        # OTHER method is a typo too (it would silently do nothing here).
        valid = {'method', 'sweep_coord'} | TWIN_KNOBS | method.knobs
        unknown = set(kwargs) - valid
        if unknown:
            raise ValueError(
                f"Unknown keyword argument(s) {sorted(unknown)} for "
                f"{type(self).__name__} with method={method.name!r}; "
                f"valid: {sorted(valid)}"
            )
        if method.requires_shots and 'shot_idx' not in dataset.coords:
            raise ValueError(
                f"method={method.name!r} needs per-shot data: no 'shot_idx' "
                f"coordinate in the dataset. Averaged acquisition is analysed "
                f"with method='average'."
            )
        slice_knobs = {k: kwargs[k] for k in method.knobs if k in kwargs}

        sweep_values = np.asarray(dataset.coords[coord].values)

        collected: Dict[str, List[Optional[np.ndarray]]] = {
            key: [] for key in method.slice_keys
        }
        failed_list: List[bool] = []
        for val in sweep_values:
            subdata = dataset.sel({coord: val})
            try:
                I, Q = state_iq_arrays(subdata)
                summary = method.reduce(I, Q, **slice_knobs)
                for key in method.slice_keys:
                    collected[key].append(np.asarray(summary[key], dtype=float))
                failed_list.append(False)
            except Exception:
                for key in method.slice_keys:
                    collected[key].append(None)
                failed_list.append(True)

        # Common shapes come from the first successful slice, then stack
        # (filling failed slices with NaN of the right shape).
        results: Dict[str, Any] = {
            'sweep_coord': coord,
            'sweep_values': sweep_values,
            'method': method.name,
            'metric': method.metric,
            'failed': np.asarray(failed_list, dtype=bool),
        }
        for key in method.slice_keys:
            results[key] = self._stack(collected[key])
        results['separation'] = self._separation_curve(results.get('mean'))
        method.derive(results)
        self._set_best(results, method, **kwargs)
        missing = [k for k in self.COMMON_KEYS if k not in results]
        if missing:
            raise ValueError(
                f"method={method.name!r} did not produce the common keys "
                f"{missing} — every method must (two-tier result contract)."
            )

        # the optional companion scale. Indexed identically to sweep_values, so the
        # answer needs a lookup, not an interpolation. Keys are absent entirely when
        # it is not drawable, so consumers test with `if key in results`.
        twin = twin_values(dataset, coord, kwargs.get('twin_coord', self.twin_coord))
        if twin is not None:
            results['twin_values'] = twin
            results['twin_label'] = str(
                kwargs.get('twin_label', self.twin_label)
                or kwargs.get('twin_coord', self.twin_coord)
            )
            idx = results.get('best_index')
            results['best_twin_value'] = (
                float(twin[idx]) if idx is not None else None
            )
        return results

    @staticmethod
    def _stack(items: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Stack a list of equal-shaped arrays, substituting NaN arrays for any
        ``None`` (failed slice). Returns ``None`` if every entry failed."""
        shape = next((a.shape for a in items if a is not None), None)
        if shape is None:
            return None
        filled = [a if a is not None else np.full(shape, np.nan) for a in items]
        return np.stack(filled, axis=0)

    @staticmethod
    def _separation_curve(mean: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Per-sweep centre separation ``|center₁ − center₀|`` (raw IQ units) —
        COMMON to every method, since every method locates centres. ``None``
        when centres are unavailable or there are fewer than two; failed slices
        (NaN centres) yield NaN."""
        if mean is None or mean.shape[1] < 2:
            return None
        return np.linalg.norm(mean[:, 0, :] - mean[:, 1, :], axis=1)  # (S,)

    @staticmethod
    def _metric_curve(results: Dict[str, Any]) -> Optional[np.ndarray]:
        """The per-sweep curve the best point maximises, named by the method."""
        return results.get(results['metric'])

    # --- best-point selection (overridable) ---------------------------
    def _set_best(
        self, results: Dict[str, Any], strategy: ReadoutFidelityMethod, **kwargs
    ) -> None:
        """Populate best_index / best_sweep_value / best_metric / success, plus
        the named ``best_fidelity`` / ``best_separation`` twins for whichever
        curves this method produced.

        ``strategy`` is the resolved method object; it cannot be called
        ``method`` because ``kwargs`` still carries the caller's method NAME."""
        idx = self._select_best_index(results, **kwargs)
        # the `best_<curve>` twins exist for every curve this method HAS (the
        # key, not a finite value) — a dead sweep answers None, never nothing
        curves = [key for key in ('fidelity', 'separation') if key in results]
        if idx is None:
            results.update(best_index=None, best_sweep_value=None,
                           best_metric=None, success=False)
            for key in curves:
                results[f'best_{key}'] = None
            return
        best = float(self._metric_curve(results)[idx])
        ok = self._selection_ok(results, idx, **kwargs)
        results.update(
            best_index=int(idx),
            best_sweep_value=float(results['sweep_values'][idx]),
            best_metric=best,
            success=bool(ok and strategy.metric_ok(best, self.fidelity_floor)),
        )
        # the answer in every curve this method has, so a downstream record can
        # state the metric it optimised AND the separation it settled for
        for key in curves:
            curve = results[key]
            results[f'best_{key}'] = float(curve[idx]) if curve is not None else None

    def _select_best_index(self, results: Dict[str, Any], **kwargs) -> Optional[int]:
        """Index of the metric-maximising sweep point (NaN-safe), or ``None``
        when no point yielded a finite metric value."""
        metric = self._metric_curve(results)
        if metric is None or not np.any(np.isfinite(metric)):
            return None
        return int(np.nanargmax(metric))

    def _selection_ok(self, results: Dict[str, Any], idx: int, **kwargs) -> bool:
        """Whether the selected point satisfies any subclass constraint. Base
        imposes none (always True)."""
        return True

    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the small answer — the sweep axis, the scalar curves, and the
        chosen best point — and drop the bulky per-slice arrays (mean, p_outlier,
        norm_res, gaussian_norms, direct_counts), which live in the plot data."""
        keep = (
            'sweep_coord', 'sweep_values', 'method', 'metric',
            'fidelity', 'snr', 'separation',
            'best_index', 'best_sweep_value', 'best_metric', 'best_fidelity',
            'best_separation', 'success',
        )
        metadata = {k: results.get(k) for k in keep}
        # the answer in the companion scale rides along; the per-point twin array
        # stays out, like the other bulky per-slice arrays
        for key in ('twin_label', 'best_twin_value'):
            if key in results:
                metadata[key] = results[key]
        return metadata

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """Assemble the per-sweep summary curves into one self-sufficient Dataset
        so the figures redraw from the saved ``*_plotdata.nc`` alone. The swept
        coordinate name is preserved both as the dimension name and in
        ``attrs['sweep_coord']``; the chosen best point rides along in attrs."""
        coord = results['sweep_coord']
        sweep = np.asarray(results['sweep_values'])

        data_vars: Dict[str, Any] = {}
        coords: Dict[str, Any] = {coord: sweep}

        # `separation` is COMMON, so it is always carried — NaN when every slice
        # failed, never absent (the merged figure must still draw)
        separation = results.get('separation')
        data_vars['separation'] = (
            coord,
            np.full(sweep.shape, np.nan) if separation is None
            else np.asarray(separation, dtype=float),
        )
        # the method-owned 1-D curves; absent ones are simply not there, and
        # each figure gates on what it needs
        for key in ('std', 'fidelity', 'snr'):
            curve = results.get(key)
            if curve is not None:
                data_vars[key] = (coord, np.asarray(curve, dtype=float))

        mean = results.get('mean')
        if mean is not None:
            coords['center'] = np.arange(mean.shape[1])
            coords['iq'] = ['I', 'Q']
            data_vars['mean'] = ([coord, 'center', 'iq'], mean)

        p_outlier = results.get('p_outlier')
        if p_outlier is not None:
            coords['prepared_state'] = np.arange(p_outlier.shape[1])
            data_vars['p_outlier'] = ([coord, 'prepared_state'], p_outlier)

        norm_res = results.get('norm_res')
        if norm_res is not None:
            coords.setdefault('prepared_state', np.arange(norm_res.shape[1]))
            data_vars['norm_res'] = ([coord, 'prepared_state'], norm_res)

        gnorms = results.get('gaussian_norms')
        if gnorms is not None:
            coords.setdefault('prepared_state', np.arange(gnorms.shape[1]))
            coords['gauss'] = np.arange(gnorms.shape[2])
            data_vars['gaussian_norms'] = ([coord, 'prepared_state', 'gauss'], gnorms)

        dcounts = results.get('direct_counts')
        if dcounts is not None:
            coords.setdefault('prepared_state', np.arange(dcounts.shape[1]))
            coords['count'] = np.arange(dcounts.shape[2])
            data_vars['direct_counts'] = ([coord, 'prepared_state', 'count'], dcounts)

        attrs: Dict[str, Any] = {
            'sweep_coord': coord,
            'method': str(results['method']),
            'metric': str(results['metric']),
        }
        if results.get('best_sweep_value') is not None:
            attrs['best_sweep_value'] = float(results['best_sweep_value'])
            attrs['best_metric'] = float(results['best_metric'])
            if results.get('best_fidelity') is not None:
                attrs['best_fidelity'] = float(results['best_fidelity'])
        # the companion scale + its label, so every sweep figure can draw the
        # secondary axis from plot_data ALONE (the self-enforcing rule)
        if results.get('twin_values') is not None:
            data_vars['twin'] = (coord, np.asarray(results['twin_values'], dtype=float))
            attrs['twin_label'] = str(results.get('twin_label', ''))
            if results.get('best_twin_value') is not None:
                attrs['best_twin_value'] = float(results['best_twin_value'])
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the readout-fidelity sweep figures, drawing only from
        ``plot_data``; rebuild it when called outside ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)

        # the centre separation and the blob width are the SAME IQ quantity —
        # one axes, so "is the separation outgrowing the width?" is read, not
        # inferred. The average method has no width and draws separation alone.
        figs: Dict[str, plt.Figure] = {
            'separation': plot_separation_vs_sweep(plot_data),
        }
        if 'snr' in plot_data:
            figs['snr'] = plot_snr_vs_sweep(plot_data)
        if 'p_outlier' in plot_data:
            figs['outlier'] = plot_outlier_vs_sweep(plot_data)
        if 'mean' in plot_data:
            figs['means_on_IQ'] = plot_means_on_iq_plane(plot_data)
        if 'norm_res' in plot_data:
            figs['norm_res'] = plot_norm_res_vs_sweep(plot_data)
        if any(v in plot_data for v in ('fidelity', 'direct_counts', 'gaussian_norms')):
            figs['fidelity'] = plot_fidelity_vs_sweep(plot_data)
        return figs


class ReadoutPowerFidelityEstimator(ReadoutFidelityEstimator):
    """Readout fidelity swept over readout amplitude (``amp_prefactor``).

    ``best_sweep_value`` is the optimal **amp_prefactor** — a multiplier on the
    current readout-pulse amplitude. The optimum is the metric-maximising point
    among amplitudes that keep the outlier population in check: pass
    ``outliers_threshold`` (e.g. 0.98) and only points whose in-distribution
    fraction ``1 - max_k p_outlier`` meets it are eligible; if none qualify the
    global maximum is returned with ``success=False``.

    That constraint is GMM-only — ``p_outlier`` is a product of the mixture fit.
    With ``method="average"`` the metric is the bare centre separation, which
    keeps growing with amplitude past the onset of measurement-induced
    transitions: use it as the fast coarse scan, not as the final amplitude
    calibration.

    Ported from qcat ``readout_power.ROFidelityPower`` (its linear mean-drift refit
    ``fit_means_vs_amp_prefactor`` is intentionally not ported — see MIGRATION.md).
    """
    estimator_name = "readout_power_fidelity"
    sweep_coord = "amp_prefactor"

    def _candidate_mask(self, results: Dict[str, Any], **kwargs) -> np.ndarray:
        """Boolean mask of sweep points allowed by ``outliers_threshold``: a point
        qualifies when its in-distribution fraction ``1 - max_k p_outlier`` is at
        least the threshold. With no threshold — or a method that has no
        p_outlier — all finite-metric points pass."""
        finite = np.isfinite(self._metric_curve(results))
        threshold = kwargs.get('outliers_threshold')
        p_outlier = results.get('p_outlier')
        if threshold is None or p_outlier is None:
            return finite
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows -> NaN
            in_dist = 1.0 - np.nanmax(p_outlier, axis=1)  # (S,)
        return finite & (in_dist >= threshold)

    def _select_best_index(self, results: Dict[str, Any], **kwargs) -> Optional[int]:
        metric = self._metric_curve(results)
        if metric is None or not np.any(np.isfinite(metric)):
            return None
        mask = self._candidate_mask(results, **kwargs)
        if np.any(mask):
            return int(np.argmax(np.where(mask, metric, -np.inf)))
        return int(np.nanargmax(metric))  # constraint unmet -> global best

    def _selection_ok(self, results: Dict[str, Any], idx: int, **kwargs) -> bool:
        return bool(self._candidate_mask(results, **kwargs)[idx])


class ReadoutFreqFidelityEstimator(ReadoutFidelityEstimator):
    """Readout fidelity swept over readout frequency (``frequency``).

    ``best_sweep_value`` is the optimal **detuning** (Hz), expressed relative to the
    current readout IF the sweep was centred on; the consuming node maps it onto an
    absolute readout frequency. The optimum is simply the metric-maximising point
    (fidelity under ``method="gmm"``, centre separation under ``"average"`` —
    both peak at the best detuning).

    Also extracts the dressed resonator frequencies for states |0> and |1>
    (``detuning_dress0``, ``detuning_dress1``) and dispersive shift ``chi``
    from the transmission response dips of |g> and |e>.

    Ported from qcat ``readout_freq.ROFidelityFreq``.
    """
    estimator_name = "readout_freq_fidelity"
    sweep_coord = "frequency"

    @staticmethod
    def _find_dip(sweep: np.ndarray, signal: np.ndarray) -> Optional[float]:
        """Find the dip (minimum) of signal vs sweep with sub-bin parabolic refinement."""
        if sweep is None or signal is None or len(sweep) == 0:
            return None
        finite = np.isfinite(signal)
        if not np.any(finite):
            return None
        idx = int(np.nanargmin(signal))
        x0 = float(sweep[idx])
        if 0 < idx < len(signal) - 1:
            x_pts = sweep[idx - 1 : idx + 2]
            y_pts = signal[idx - 1 : idx + 2]
            if np.all(np.isfinite(y_pts)):
                try:
                    poly = np.polyfit(x_pts, y_pts, 2)
                    if poly[0] > 0:  # concave up (minimum)
                        x_min = -poly[1] / (2.0 * poly[0])
                        if min(x_pts[0], x_pts[2]) <= x_min <= max(x_pts[0], x_pts[2]):
                            x0 = float(x_min)
                except Exception:
                    pass
        return x0

    def _extract_dressed_dips(
        self, results: Dict[str, Any], dataset: xr.Dataset, **kwargs
    ) -> None:
        mean = results.get("mean")
        sweep = results.get("sweep_values")
        if mean is None or sweep is None or mean.shape[1] < 2:
            results["detuning_dress0"] = None
            results["detuning_dress1"] = None
            results["chi"] = None
            return

        iq_abs_0 = np.linalg.norm(mean[:, 0, :], axis=-1)
        iq_abs_1 = np.linalg.norm(mean[:, 1, :], axis=-1)

        dip0 = self._find_dip(sweep, iq_abs_0)
        dip1 = self._find_dip(sweep, iq_abs_1)

        results["detuning_dress0"] = dip0
        results["detuning_dress1"] = dip1

        if dip0 is not None and dip1 is not None and np.isfinite(dip0) and np.isfinite(dip1):
            results["chi"] = float((dip0 - dip1) / 2.0)
        else:
            results["chi"] = None

        twin = results.get("twin_values")
        if twin is not None and len(twin) == len(sweep) and len(sweep) > 1:
            if dip0 is not None and np.isfinite(dip0):
                results["full_freq_dress0"] = float(np.interp(dip0, sweep, twin))
            if dip1 is not None and np.isfinite(dip1):
                results["full_freq_dress1"] = float(np.interp(dip1, sweep, twin))

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        results = super().extract_parameters(dataset, **kwargs)
        self._extract_dressed_dips(results, dataset, **kwargs)
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        metadata = super().extract_metadata(results)
        for key in (
            "detuning_dress0",
            "detuning_dress1",
            "chi",
            "full_freq_dress0",
            "full_freq_dress1",
        ):
            if key in results:
                val = results[key]
                metadata[key] = float(val) if val is not None and np.isfinite(val) else None
        return metadata

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        plot_data = super().build_plot_data(dataset, results, **kwargs)
        for key in (
            "detuning_dress0",
            "detuning_dress1",
            "chi",
            "full_freq_dress0",
            "full_freq_dress1",
        ):
            val = results.get(key)
            if val is not None and np.isfinite(val):
                plot_data.attrs[key] = float(val)
        return plot_data

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        figs = super().generate_figures(dataset, results, plot_data=plot_data, **kwargs)
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        if "mean" in plot_data:
            figs["response"] = plot_response_vs_frequency(plot_data)
        return figs

