"""Crosstalk Compensated Deterministic Benchmark Estimator.

Analyzes deterministic repeated gate benchmark comparing three conditions:
1. Isolated: Probe qubit executes N target gates alone.
2. Simultaneous: Probe qubit and drive qubit execute N target gates concurrently.
3. Compensated: Probe and drive execute N gates concurrently with active cancellation.

Also supports calibration mode for scanning 2D (cancel_amp, init_phase) grids
with an idle probe qubit in |0> to locate the cancellation minimum.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator


def _smooth_2d(z: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Smooth 2D array lightly to suppress single-shot readout spikes."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(z, sigma=sigma)
    except Exception:
        pad = np.pad(z, 1, mode="edge")
        return (
            pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
            pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
            pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
        ) / 9.0


def _fit_quadratic_2d_extremum(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_grid: np.ndarray,
    mode: str = "min",
    smooth_sigma: float = 0.8,
    patch_radius: int = 2,
) -> Dict[str, Any]:
    """Fit a 2D bivariate quadratic surface to locate continuous extremum with noise rejection.

    Model: f(dx, dy) = c0 + c1*dx + c2*dy + c3*dx^2 + c4*dy^2 + c5*dx*dy
    """
    Nx, Ny = z_grid.shape
    z_clean = np.nan_to_num(z_grid, nan=float(np.nanmean(z_grid)))

    # 1. Robust center finding using light Gaussian smoothing to prevent outlier trapping
    z_smooth = _smooth_2d(z_clean, sigma=smooth_sigma) if smooth_sigma > 0 else z_clean
    if mode == "min":
        coarse_idx = np.unravel_index(np.argmin(z_smooth), z_smooth.shape)
    else:
        coarse_idx = np.unravel_index(np.argmax(z_smooth), z_smooth.shape)

    i0, j0 = int(coarse_idx[0]), int(coarse_idx[1])
    coarse_x = float(x_coords[i0])
    coarse_y = float(y_coords[j0])

    d_x = float(x_coords[1] - x_coords[0]) if Nx > 1 else 1.0
    d_y = float(y_coords[1] - y_coords[0]) if Ny > 1 else 1.0

    # Check boundaries for local patch
    r_x = min(patch_radius, i0, Nx - 1 - i0)
    r_y = min(patch_radius, j0, Ny - 1 - j0)

    if r_x < 1 or r_y < 1:
        return {
            "opt_x": coarse_x,
            "opt_y": coarse_y,
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "coarse_idx": (i0, j0),
            "delta_x": 0.0,
            "delta_y": 0.0,
            "is_interpolated": False,
        }

    # Extract patch coordinates and values centered at (i0, j0)
    A_rows = []
    b_vals = []
    for di in range(-r_x, r_x + 1):
        for dj in range(-r_y, r_y + 1):
            dx = float(x_coords[i0 + di] - coarse_x)
            dy = float(y_coords[j0 + dj] - coarse_y)
            A_rows.append([1.0, dx, dy, dx**2, dy**2, dx * dy])
            b_vals.append(float(z_clean[i0 + di, j0 + dj]))

    A_mat = np.asarray(A_rows, dtype=float)
    b_vec = np.asarray(b_vals, dtype=float)

    c, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    g = np.array([c[1], c[2]], dtype=float)
    H = np.array([[2.0 * c[3], c[5]], [c[5], 2.0 * c[4]]], dtype=float)

    eigvals = np.linalg.eigvalsh(H)
    if mode == "min":
        valid_curvature = bool(np.all(eigvals > 1e-9))
    else:
        valid_curvature = bool(np.all(eigvals < -1e-9))

    if not valid_curvature:
        return {
            "opt_x": coarse_x,
            "opt_y": coarse_y,
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "coarse_idx": (i0, j0),
            "delta_x": 0.0,
            "delta_y": 0.0,
            "is_interpolated": False,
        }

    delta = -np.linalg.solve(H, g)
    delta_x = float(np.clip(delta[0], -abs(d_x), abs(d_x)))
    delta_y = float(np.clip(delta[1], -abs(d_y), abs(d_y)))

    return {
        "opt_x": coarse_x + delta_x,
        "opt_y": coarse_y + delta_y,
        "coarse_x": coarse_x,
        "coarse_y": coarse_y,
        "coarse_idx": (i0, j0),
        "delta_x": delta_x,
        "delta_y": delta_y,
        "is_interpolated": True,
        "hessian": H.tolist(),
        "coefficients": c.tolist(),
    }


class CrosstalkCompensatedBenchmarkEstimator(BaseEstimator):
    """Analyze deterministic crosstalk benchmark and 2D calibration."""

    estimator_name = "crosstalk_compensated_benchmark"

    def _check_data(self, dataset: xr.Dataset, **kwargs) -> None:
        if not any(var in dataset.data_vars for var in ("I", "state", "signal", "population")):
            raise ValueError(
                "CrosstalkCompensatedBenchmarkEstimator requires an 'I', 'state', 'signal', or 'population' data variable."
            )
        if "cancel_amp" in dataset.coords and "init_phase" in dataset.coords:
            return
        elif "phase_rate" in dataset.coords:
            return
        elif "condition" in dataset.coords and "repetitions" in dataset.coords:
            return
        else:
            raise ValueError(
                "CrosstalkCompensatedBenchmarkEstimator dataset must contain either "
                "('cancel_amp', 'init_phase') coordinates for calibrate mode, "
                "('phase_rate',) coordinate for calibrate_phase_rate mode, or "
                "('condition', 'repetitions') coordinates for benchmark mode."
            )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        self._check_data(dataset)
        if "cancel_amp" in dataset.coords and "init_phase" in dataset.coords:
            mode = "calibrate"
        else:
            mode = "benchmark"

        var_name = (
            "population"
            if "population" in dataset.data_vars
            else ("state" if "state" in dataset.data_vars else ("I" if "I" in dataset.data_vars else "signal"))
        )

        if mode == "calibrate":
            return self._extract_calibration(dataset, var_name)
        return self._extract_benchmark(dataset, var_name)

    def _extract_calibration(
        self, dataset: xr.Dataset, var_name: str
    ) -> Dict[str, Any]:
        amps = np.asarray(dataset.coords["cancel_amp"].values, dtype=float)
        phases = np.asarray(dataset.coords["init_phase"].values, dtype=float)

        da = dataset[var_name]
        p_grid = np.squeeze(da.values)

        fit_res = _fit_quadratic_2d_extremum(
            amps, phases, p_grid, mode="min", smooth_sigma=0.8, patch_radius=2
        )

        opt_amp = fit_res["opt_x"]
        opt_phase = fit_res["opt_y"]
        best_amp_coarse = fit_res["coarse_x"]
        best_phase_coarse = fit_res["coarse_y"]
        coarse_idx = fit_res["coarse_idx"]
        min_p1 = float(p_grid[coarse_idx])

        return {
            "mode": "calibrate",
            "optimal_cancel_amp": float(opt_amp),
            "optimal_init_phase": float(opt_phase),
            "optimal_init_phase_deg": float(np.degrees(opt_phase)),
            "min_residual_p1": float(min_p1),
            "success": True,
            "suggested_updates": {
                "cancel_amp": float(opt_amp),
                "init_phase": float(opt_phase),
            },
        }

    def _extract_benchmark(
        self, dataset: xr.Dataset, var_name: str
    ) -> Dict[str, Any]:
        cond_da = dataset.coords["condition"].values
        reps = np.asarray(dataset.coords["repetitions"].values, dtype=int)
        da = dataset[var_name]

        condition_results: Dict[str, Any] = {}
        for cond in cond_da:
            cond_str = str(cond)
            sub_da = da.sel(condition=cond)
            vals = np.squeeze(sub_da.values)

            # Calculate average error across repetition points
            mean_err = float(np.mean(vals))
            max_err = float(np.max(vals))
            condition_results[cond_str] = {
                "mean_error": mean_err,
                "max_error": max_err,
                "values": vals.tolist(),
            }

        iso_err = condition_results.get("isolated", {}).get("mean_error", 0.0)
        sim_err = condition_results.get("simultaneous", {}).get("mean_error", 0.0)
        comp_err = condition_results.get("compensated", {}).get("mean_error", 0.0)

        crosstalk_penalty = max(0.0, sim_err - iso_err)
        mitigation_fraction = 0.0
        if crosstalk_penalty > 1e-6:
            mitigated = max(0.0, sim_err - comp_err)
            mitigation_fraction = float(np.clip(mitigated / crosstalk_penalty, 0.0, 1.0))

        return {
            "mode": "benchmark",
            "conditions": condition_results,
            "mean_error_isolated": iso_err,
            "mean_error_simultaneous": sim_err,
            "mean_error_compensated": comp_err,
            "crosstalk_penalty": crosstalk_penalty,
            "mitigation_fraction": mitigation_fraction,
            "success": True,
        }

    def generate_figures(
        self,
        dataset: xr.Dataset,
        fit_results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        mode = fit_results.get("mode", "benchmark")
        var_name = (
            "population"
            if "population" in dataset.data_vars
            else ("state" if "state" in dataset.data_vars else ("I" if "I" in dataset.data_vars else "signal"))
        )

        figures: Dict[str, plt.Figure] = {}

        if mode == "calibrate":
            history = None
            if "stage_history" in dataset.attrs:
                try:
                    import json
                    raw_hist = dataset.attrs["stage_history"]
                    history = json.loads(raw_hist) if isinstance(raw_hist, str) else raw_hist
                except Exception:
                    history = None

            if history and len(history) > 1:
                # 1. Multi-panel side-by-side progression plot
                n_stages = len(history)
                fig_summary, axes = plt.subplots(1, n_stages, figsize=(5.0 * n_stages, 4.5), squeeze=False)
                for idx, s in enumerate(history):
                    ax = axes[0, idx]
                    s_amps = np.asarray(s.get("cancel_amps", []))
                    s_phases = np.asarray(s.get("init_phases", []))
                    s_p = np.asarray(s.get("p_vals", []))
                    if len(s_amps) > 0 and len(s_phases) > 0 and s_p.size > 0:
                        im = ax.imshow(
                            s_p.T,
                            origin="lower",
                            extent=[s_amps[0], s_amps[-1], s_phases[0], s_phases[-1]],
                            aspect="auto",
                            cmap="viridis",
                        )
                        fig_summary.colorbar(im, ax=ax, label="Probe Excitation")
                    best_a = s.get("best_cancel_amp")
                    best_p = s.get("best_init_phase")
                    if idx == n_stages - 1 and fit_results.get("optimal_cancel_amp") is not None:
                        best_a = fit_results["optimal_cancel_amp"]
                        best_p = fit_results["optimal_init_phase"]
                    if best_a is not None and best_p is not None:
                        ax.plot(best_a, best_p, "r*", markersize=14, label=f"Opt: ({best_a:.4f}, {best_p:.3f})")
                        ax.legend(loc="upper right", fontsize=8)
                    ax.set_title(f"Stage {s.get('stage', idx + 1)}: N={s.get('repetitions', '?')}")
                    ax.set_xlabel("Cancel Amp Scale")
                    if idx == 0:
                        ax.set_ylabel("Initial Phase (rad)")
                fig_summary.tight_layout()
                figures["summary"] = fig_summary

                # 2. Individual stage detailed plots
                for idx, s in enumerate(history):
                    stage_num = s.get("stage", idx + 1)
                    fig_st, ax_st = plt.subplots(figsize=(7, 5))
                    s_amps = np.asarray(s.get("cancel_amps", []))
                    s_phases = np.asarray(s.get("init_phases", []))
                    s_p = np.asarray(s.get("p_vals", []))
                    if len(s_amps) > 0 and len(s_phases) > 0 and s_p.size > 0:
                        im_st = ax_st.imshow(
                            s_p.T,
                            origin="lower",
                            extent=[s_amps[0], s_amps[-1], s_phases[0], s_phases[-1]],
                            aspect="auto",
                            cmap="viridis",
                        )
                        fig_st.colorbar(im_st, ax=ax_st, label="Probe Excitation P(|1|)")
                    best_a = s.get("best_cancel_amp")
                    best_p = s.get("best_init_phase")
                    if idx == len(history) - 1 and fit_results.get("optimal_cancel_amp") is not None:
                        best_a = fit_results["optimal_cancel_amp"]
                        best_p = fit_results["optimal_init_phase"]
                    if best_a is not None and best_p is not None:
                        ax_st.plot(best_a, best_p, "r*", markersize=14, label=f"Opt: ({best_a:.4f}, {best_p:.3f} rad)")
                        ax_st.legend(loc="upper right")
                    ax_st.set_xlabel("Cancel Amplitude Scale")
                    ax_st.set_ylabel("Initial Phase (rad)")
                    ax_st.set_title(f"Crosstalk Calibration Stage {stage_num} (N={s.get('repetitions', '?')})")
                    fig_st.tight_layout()
                    figures[f"stage_{stage_num}"] = fig_st

            else:
                fig, ax = plt.subplots(figsize=(8, 5))
                amps = np.asarray(dataset.coords["cancel_amp"].values, dtype=float)
                phases = np.asarray(dataset.coords["init_phase"].values, dtype=float)
                p_grid = np.squeeze(dataset[var_name].values)

                im = ax.imshow(
                    p_grid.T,
                    origin="lower",
                    extent=[amps[0], amps[-1], phases[0], phases[-1]],
                    aspect="auto",
                    cmap="viridis",
                )
                fig.colorbar(im, ax=ax, label="Probe Excitation P(|1|)")
                ax.set_xlabel("Cancel Amplitude Scale")
                ax.set_ylabel("Initial Phase (rad)")
                ax.set_title("Crosstalk Cancellation 2D Calibration")

                opt_a = fit_results.get("optimal_cancel_amp")
                opt_p = fit_results.get("optimal_init_phase")
                if opt_a is not None and opt_p is not None:
                    ax.plot(opt_a, opt_p, "r*", markersize=14, label=f"Opt: ({opt_a:.3f}, {opt_p:.2f} rad)")
                    ax.legend(loc="upper right")
                fig.tight_layout()
                figures["summary"] = fig

        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            reps = np.asarray(dataset.coords["repetitions"].values, dtype=int)
            da = dataset[var_name]

            colors = {"isolated": "blue", "simultaneous": "red", "compensated": "green"}
            for cond in dataset.coords["condition"].values:
                cond_str = str(cond)
                y = np.squeeze(da.sel(condition=cond).values)
                color = colors.get(cond_str, "black")
                ax.plot(reps, y, "o-", color=color, label=f"{cond_str.capitalize()}")

            ax.set_xlabel("Gate Repetitions N")
            ax.set_ylabel("Probe Error / State")
            ax.set_title("Crosstalk Compensated Deterministic Benchmark")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            figures["summary"] = fig

        return figures

