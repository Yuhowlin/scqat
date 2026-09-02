"""Crosstalk Compensated Simultaneous SQRB Estimator.

Analyzes Single Qubit Randomized Benchmarking (SQRB) comparing three conditions:
1. Isolated: Probe qubit runs SQRB alone.
2. Simultaneous: Probe qubit and drive qubit run SQRB concurrently without compensation.
3. Compensated: Probe qubit and drive qubit run SQRB concurrently with active compensation.

Also supports calibration mode for scanning 2D (cancel_amp, init_phase) grids.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_powerlaw_base import FitBasePowerLaw


class CrosstalkCompensatedSQRBEstimator(BaseEstimator):
    """Fit 3-condition SQRB decay curves and quantify crosstalk mitigation."""

    estimator_name = "crosstalk_compensated_sqrb"

    def _check_data(self, dataset: xr.Dataset, **kwargs) -> None:
        if not any(var in dataset.data_vars for var in ("I", "state", "signal", "population")):
            raise ValueError(
                "CrosstalkCompensatedSQRBEstimator requires an 'I', 'state', 'signal', or 'population' data variable."
            )
        if "cancel_amp" in dataset.coords and "init_phase" in dataset.coords:
            return
        elif "condition" in dataset.coords and "depth" in dataset.coords:
            return
        else:
            raise ValueError(
                "CrosstalkCompensatedSQRBEstimator dataset must contain either "
                "('cancel_amp', 'init_phase') coordinates for calibrate mode, or "
                "('condition', 'depth') coordinates for benchmark mode."
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
        if "sequence_idx" in da.dims:
            p_grid = da.mean(dim="sequence_idx").values
        else:
            p_grid = da.values
        p_grid = np.squeeze(p_grid)

        # Convert excited state population P1 to ground state survival P0 = 1 - P1
        if var_name in ("state", "population"):
            p_grid = 1.0 - p_grid

        # Locate coarse maximum on 2D grid
        max_idx = np.unravel_index(np.argmax(p_grid), p_grid.shape)
        best_amp_coarse = amps[max_idx[0]]
        best_phase_coarse = phases[max_idx[1]]
        max_p0 = float(p_grid[max_idx])

        # Sub-grid 2D quadratic interpolation if inside boundary
        opt_amp = best_amp_coarse
        opt_phase = best_phase_coarse
        i, j = max_idx
        if 0 < i < len(amps) - 1 and 0 < j < len(phases) - 1:
            # 1D vertex parabola on amp axis
            y1, y2, y3 = p_grid[i - 1, j], p_grid[i, j], p_grid[i + 1, j]
            d_amp = amps[1] - amps[0]
            if (2 * y2 - y1 - y3) > 1e-12:
                delta_i = float(np.clip(0.5 * (y1 - y3) / (y1 - 2 * y2 + y3), -0.5, 0.5))
                opt_amp = amps[i] + delta_i * d_amp

            # 1D vertex parabola on phase axis
            z1, z2, z3 = p_grid[i, j - 1], p_grid[i, j], p_grid[i, j + 1]
            d_phase = phases[1] - phases[0]
            if (2 * z2 - z1 - z3) > 1e-12:
                delta_j = float(np.clip(0.5 * (z1 - z3) / (z1 - 2 * z2 + z3), -0.5, 0.5))
                opt_phase = phases[j] + delta_j * d_phase

        return {
            "mode": "calibrate",
            "optimal_cancel_amp": float(opt_amp),
            "optimal_init_phase_rad": float(opt_phase),
            "optimal_init_phase_deg": float(np.degrees(opt_phase)),
            "max_population": max_p0,
            "cancel_amps": amps.tolist(),
            "init_phases": phases.tolist(),
            "population_grid": p_grid.tolist(),
            "success": bool(np.isfinite(max_p0) and max_p0 > 0.5),
        }

    def _extract_benchmark(
        self, dataset: xr.Dataset, var_name: str
    ) -> Dict[str, Any]:
        conditions = [str(c) for c in dataset.coords["condition"].values]
        depths = np.asarray(dataset.coords["depth"].values, dtype=float)

        results: Dict[str, Any] = {
            "mode": "benchmark",
            "conditions": conditions,
            "depths": depths.tolist(),
            "per_condition": {},
            "best_fit": {},
        }

        all_success = True
        avg_gates_per_clifford = 1.875

        for cond in conditions:
            da_cond = dataset[var_name].sel(condition=cond)
            if "sequence_idx" in da_cond.dims:
                da_avg = da_cond.mean(dim="sequence_idx").squeeze().rename({"depth": "x"})
            else:
                da_avg = da_cond.squeeze().rename({"depth": "x"})

            fit_res = FitBasePowerLaw(da_avg).fit()
            alpha = float(fit_res.params["base"].value) if fit_res.success else np.nan
            err_clifford = 0.5 * (1.0 - alpha) if np.isfinite(alpha) else np.nan
            err_gate = (
                err_clifford / avg_gates_per_clifford
                if np.isfinite(err_clifford)
                else np.nan
            )
            gate_fid = 1.0 - err_gate if np.isfinite(err_gate) else np.nan

            cond_ok = bool(fit_res.success) and (0.0 < alpha <= 1.0)
            if not cond_ok:
                all_success = False

            results["per_condition"][cond] = {
                "alpha": alpha,
                "alpha_stderr": float(fit_res.params["base"].stderr or np.nan),
                "amplitude": float(fit_res.params["a"].value) if fit_res.success else np.nan,
                "offset": float(fit_res.params["c"].value) if fit_res.success else np.nan,
                "error_per_clifford": err_clifford,
                "error_per_gate": err_gate,
                "gate_fidelity": gate_fid,
                "success": cond_ok,
            }
            results["best_fit"][cond] = (
                np.asarray(fit_res.best_fit, dtype=float).tolist()
                if fit_res.success
                else [np.nan] * len(depths)
            )

        # Cross-condition crosstalk metrics
        conds_map = results["per_condition"]
        r_iso = conds_map.get("isolated", {}).get("error_per_clifford", np.nan)
        r_sim = conds_map.get("simultaneous", {}).get("error_per_clifford", np.nan)
        r_comp = conds_map.get("compensated", {}).get("error_per_clifford", np.nan)

        f_iso = conds_map.get("isolated", {}).get("gate_fidelity", np.nan)
        f_sim = conds_map.get("simultaneous", {}).get("gate_fidelity", np.nan)
        f_comp = conds_map.get("compensated", {}).get("gate_fidelity", np.nan)

        rg_iso = conds_map.get("isolated", {}).get("error_per_gate", np.nan)
        rg_sim = conds_map.get("simultaneous", {}).get("error_per_gate", np.nan)
        rg_comp = conds_map.get("compensated", {}).get("error_per_gate", np.nan)

        crosstalk_error_penalty = (
            r_sim - r_iso if np.isfinite(r_sim) and np.isfinite(r_iso) else np.nan
        )
        crosstalk_mitigated_error = (
            r_sim - r_comp if np.isfinite(r_sim) and np.isfinite(r_comp) else np.nan
        )

        mitigation_ratio = np.nan
        if np.isfinite(crosstalk_error_penalty) and crosstalk_error_penalty > 1e-7:
            mitigation_ratio = float(
                np.clip(crosstalk_mitigated_error / crosstalk_error_penalty, 0.0, 1.5)
            )

        results["r_isolated"] = r_iso
        results["r_simultaneous"] = r_sim
        results["r_compensated"] = r_comp
        results["gate_fidelity_isolated"] = f_iso
        results["gate_fidelity_simultaneous"] = f_sim
        results["gate_fidelity_compensated"] = f_comp
        results["error_per_gate_isolated"] = rg_iso
        results["error_per_gate_simultaneous"] = rg_sim
        results["error_per_gate_compensated"] = rg_comp
        results["gate_fidelity"] = f_comp if np.isfinite(f_comp) else f_iso
        results["error_per_gate"] = rg_comp if np.isfinite(rg_comp) else rg_iso
        results["error_per_clifford"] = r_comp if np.isfinite(r_comp) else r_iso
        results["crosstalk_error_penalty"] = crosstalk_error_penalty
        results["crosstalk_mitigated_error"] = crosstalk_mitigated_error
        results["mitigation_ratio"] = mitigation_ratio
        results["success"] = all_success

        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v
            for k, v in results.items()
            if k not in ("best_fit", "population_grid")
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        mode = results.get("mode", "benchmark")
        var_name = (
            "population"
            if "population" in dataset.data_vars
            else ("state" if "state" in dataset.data_vars else ("I" if "I" in dataset.data_vars else "signal"))
        )

        if mode == "calibrate":
            amps = np.asarray(results["cancel_amps"], dtype=float)
            phases = np.asarray(results["init_phases"], dtype=float)
            p_grid = np.asarray(results["population_grid"], dtype=float)
            return xr.Dataset(
                {"population_grid": (("cancel_amp", "init_phase"), p_grid)},
                coords={"cancel_amp": amps, "init_phase": phases},
                attrs={"optimal_cancel_amp": results["optimal_cancel_amp"],
                       "optimal_init_phase_deg": results["optimal_init_phase_deg"]},
            )

        depths = np.asarray(results["depths"], dtype=float)
        data_vars: Dict[str, Any] = {}
        for cond in results["conditions"]:
            da_cond = dataset[var_name].sel(condition=cond)
            if "sequence_idx" in da_cond.dims:
                y_avg = da_cond.mean(dim="sequence_idx").values
            else:
                y_avg = da_cond.values
            data_vars[f"signal_{cond}"] = ("depth", np.asarray(y_avg, dtype=float))
            data_vars[f"best_fit_{cond}"] = ("depth", np.asarray(results["best_fit"][cond], dtype=float))

        return xr.Dataset(
            data_vars,
            coords={"depth": depths},
            attrs={
                "gate_fidelity": results.get("gate_fidelity", np.nan),
                "error_per_gate": results.get("error_per_gate", np.nan),
                "error_per_clifford": results.get("error_per_clifford", np.nan),
                "r_isolated": results.get("r_isolated", np.nan),
                "r_simultaneous": results.get("r_simultaneous", np.nan),
                "r_compensated": results.get("r_compensated", np.nan),
                "mitigation_ratio": results.get("mitigation_ratio", np.nan),
                "success": int(bool(results.get("success", False))),
            },
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)

        mode = results.get("mode", "benchmark")
        if mode == "calibrate":
            return self._generate_calibration_figure(results)
        return self._generate_benchmark_figure(plot_data, results)

    def _generate_calibration_figure(
        self, results: Dict[str, Any]
    ) -> Dict[str, plt.Figure]:
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
        amps = np.asarray(results["cancel_amps"])
        phases_deg = np.degrees(np.asarray(results["init_phases"]))
        p_grid = np.asarray(results["population_grid"]).T  # transpose for (phases, amps)

        c = ax.contourf(amps, phases_deg, p_grid, levels=25, cmap="viridis")
        fig.colorbar(c, ax=ax, label="Survival Population P(|0>)")

        opt_a = results["optimal_cancel_amp"]
        opt_p_deg = results["optimal_init_phase_deg"]
        ax.plot(
            opt_a,
            opt_p_deg,
            marker="*",
            color="red",
            markersize=14,
            label=f"Optimal: amp={opt_a:.4f}, phase={opt_p_deg:.1f} deg",
        )

        ax.set_xlabel("Compensation Amplitude Scale (cancel_amp)")
        ax.set_ylabel("Initial Phase (deg)")
        ax.set_title("Crosstalk Compensation Calibration Map")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        return {"crosstalk_compensation_calibration": fig}

    def _generate_benchmark_figure(
        self, plot_data: xr.Dataset, results: Dict[str, Any]
    ) -> Dict[str, plt.Figure]:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        depths = np.asarray(plot_data["depth"].values, dtype=float)

        style_map = {
            "isolated": {"color": "#1f77b4", "label": "Isolated", "marker": "o"},
            "simultaneous": {"color": "#d62728", "label": "Simultaneous", "marker": "s"},
            "compensated": {"color": "#2ca02c", "label": "Compensated", "marker": "^"},
        }

        min_depth = max(1.0, float(np.min(depths)))
        max_depth = max(min_depth + 1.0, float(np.max(depths)))
        m_dense = np.geomspace(min_depth, max_depth, 200)

        for cond in results["conditions"]:
            st = style_map.get(cond, {"color": "gray", "label": cond, "marker": "."})
            sig_var = f"signal_{cond}"
            cond_res = results.get("per_condition", {}).get(cond, {})

            if sig_var in plot_data:
                ax.plot(
                    depths,
                    plot_data[sig_var],
                    st["marker"],
                    color=st["color"],
                    markersize=5,
                    label=f"{st['label']} Data (seq avg)",
                )

            if cond_res.get("success", False):
                alpha = cond_res.get("alpha", np.nan)
                a = cond_res.get("amplitude", 0.5)
                c = cond_res.get("offset", 0.5)
                f_g = cond_res.get("gate_fidelity", np.nan) * 100.0
                if np.isfinite(alpha) and np.isfinite(a) and np.isfinite(c):
                    y_dense = a * (alpha ** m_dense) + c
                    ax.plot(
                        m_dense,
                        y_dense,
                        "-",
                        color=st["color"],
                        linewidth=2,
                        label=f"{st['label']} Fit (F = {f_g:.3f}%)" if np.isfinite(f_g) else f"{st['label']} Fit",
                    )
            elif f"best_fit_{cond}" in plot_data and np.isfinite(plot_data[f"best_fit_{cond}"].values).all():
                ax.plot(
                    depths,
                    plot_data[f"best_fit_{cond}"],
                    "-",
                    color=st["color"],
                    linewidth=2,
                    label=f"{st['label']} Fit",
                )

        ax.set_xscale("log")
        ax.set_xlabel("Clifford Depth (log scale)", fontsize=11)
        ax.set_ylabel("Readout Signal (a.u.)", fontsize=11)

        f_comp = results.get("gate_fidelity_compensated", np.nan) * 100.0
        f_iso = results.get("gate_fidelity_isolated", np.nan) * 100.0
        f_sim = results.get("gate_fidelity_simultaneous", np.nan) * 100.0

        if np.isfinite(f_comp):
            title_text = (
                f"Crosstalk Compensated Simultaneous SQRB\n"
                f"Compensated Gate Fidelity: {f_comp:.3f}% (Iso: {f_iso:.3f}%, Sim: {f_sim:.3f}%)"
            )
        elif np.isfinite(f_iso):
            title_text = (
                f"Single Qubit Randomized Benchmarking (SQRB)\n"
                f"Gate Fidelity: {f_iso:.3f}%"
            )
        else:
            title_text = "Crosstalk Compensated Simultaneous SQRB"

        ax.set_title(title_text, fontsize=12, fontweight="bold")

        # Summary info box matching qubit_sqrb style
        info_lines = []
        cond_map = results.get("per_condition", {})
        for cond in ("isolated", "simultaneous", "compensated"):
            if cond in cond_map:
                c_res = cond_map[cond]
                fid = c_res.get("gate_fidelity", np.nan) * 100.0
                rg = c_res.get("error_per_gate", np.nan)
                rc = c_res.get("error_per_clifford", np.nan)
                alpha = c_res.get("alpha", np.nan)
                alpha_err = c_res.get("alpha_stderr", np.nan)
                if np.isfinite(alpha_err):
                    info_lines.append(
                        f"{cond.capitalize():12s}: F={fid:.3f}%, r_g={rg:.3e}, r_c={rc:.3e}, α={alpha:.5f}±{alpha_err:.5f}"
                    )
                else:
                    info_lines.append(
                        f"{cond.capitalize():12s}: F={fid:.3f}%, r_g={rg:.3e}, r_c={rc:.3e}, α={alpha:.5f}"
                    )

        eta = results.get("mitigation_ratio", np.nan)
        if np.isfinite(eta):
            info_lines.append(f"Mitigation Recovery: {eta * 100:.1f}%")

        if info_lines:
            ax.text(
                0.05,
                0.05,
                "\n".join(info_lines),
                transform=ax.transAxes,
                fontsize=9,
                fontfamily="monospace",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
            )

        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        fig.tight_layout()
        return {"crosstalk_compensated_sqrb": fig}

