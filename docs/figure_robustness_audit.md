# Figure-robustness audit — "raw data must always be plottable"

Scopes the follow-up migration to the contract in `CLAUDE.md` ("Raw data must
always be plottable"). **Done:** the shared mechanism `scqat/core/figures.py::render_figures`
(per-figure isolation) + the two cryoscope estimators adopt it + a "figures render
on a failed fit" regression test; `parity_switch_continuous` migrated with its
rename (and its new sibling `parity_switch_discrete` was born on `render_figures`),
both with the failed-fit test; `parametric_drive_decoherence` migrated when SCQO's
`qubit_parametric_drive_time` became its first orchestrated consumer. **This doc:**
which of the other estimators still need migrating, and in what order.

Of the estimators with `generate_figures`, only the 2 cryoscopes, the 2
parity-switch siblings and `parametric_drive_decoherence` use `render_figures`. Of
the rest, **9 are AT RISK**;
the others are effectively safe (single raw-only figure, or the raw is drawn
unconditionally and the fit overlay is guarded/NaN-safe).

## AT RISK — migrate to `render_figures`

Two failure modes: **crash-on-empty-fit** (a plotter raises on all-NaN fit data) and
**no-isolation** (a raw map/trace shares one un-isolated dict with a fit sibling, so a
sibling raise drops the raw figure too).

| estimator | #figs | risk | evidence (file:line) |
|---|---|---|---|
| **state_discrimination** | 4 | **crash-on-empty-fit + no-isolation** | `visualization.py:170` `absmax = max(abs(nanmin(residues)), abs(nanmax(residues)))` → NaN when `fit_residue` all-NaN → `pcolormesh(vmin=-absmax,vmax=absmax)`+`colorbar` (`:174-179`) raises; the raw I/Q `raw` fig is built first and lost. **The direct analog of the cryoscope bug — fix first.** |
| **charge_gate_ramsey** | 4 | no-isolation | `estimator.py:259-264` sequential; raw `raw_colormap` (`visualization.py:25`); latent unconditional `attrs['f_c']` (`:165`) |
| **ac_stark_shift** | 2 | no-isolation | `estimator.py:183-186`; raw `raw_2d` pcolormesh (`visualization.py:33`) |
| **readout_pulse_photon** | 2 | no-isolation | `estimator.py:163-166`; raw `raw_2d` (`visualization.py:31`) |
| **zz_interaction** | 2 | no-isolation | `estimator.py:140-143`; raw `raw_data` 2D colormap (`visualization.py:19`) |
| **readout_fidelity** | ≤9 | no-isolation (mass-drop) | `estimator.py:335-350`; plotters individually simple |
| **qubit_tomography** | 3 | no-isolation | `estimator.py:198-202`; fragile 3D wireframe/colorbar (`:169-179`) can drop the 2D raw; unconditional `attrs['lim_I_low'…]` (`:201-202`) |
| **ramsey** | 2 (+iq) | no-isolation (low) | `estimator.py:138-143`; both plotters guarded (`visualization.py:32` `if 'best_fit'`) — structural only |
| **qubit_decoherence** | ≤2 | no-isolation (low) | `estimator.py:183` returns a per-variable loop dict; degenerate-fit-safe — structural only |

## SAFE (no migration needed for correctness)

`qc_n_swap_amp`, `pair_swap_flux_map`, `pair_swap_chevron`, `parametric_drive_resonance`,
`parametric_drive_decoherence` (migrated — `render_figures` + a failed-fit test; its
`plot_rho11_fits` already guarded the overlay with `np.isfinite(fit[i]).any()`),
`qubit_spectroscopy_flux`, `qubit_flux_arch`, `qubit_relaxation_flux`, `qubit_echo_flux`,
`resonator_spectroscopy{,_flux,_power}`, `power_rabi`, `swap_oscillation`, `xyz_delay`,
`qubit_drag_equator`, `qubit_drag_alternating`, `qubit_deterministic_benchmarking`,
`qubit_spectroscopy`, `single_state_outlier`, `qubit_sqrb` — single raw-only figure,
or raw drawn unconditionally with a `.any()`/`if "best_fit"`/`success`-guarded fit
overlay (several even title "[FIT FAILED]").

Two SAFE-with-caveat: **`qubit_relaxation`** / **`qubit_echo`** — `plot_decay`
(`visualization.py:12-13`) plots `best_fit` + reads `attrs["t1"]`/`["t2_echo"]`
unconditionally; NaN-safe in practice (the attr is always present as NaN), but the one
spot deviating from rule 3's guard discipline — tidy when migrating.

The `+iq` estimators append `figs["iq_plane"]` after their main dict; `plot_iq_plane`
(`_iq_plane.py:73`) is raw-only, fully guarded, and only runs when `has_iq_plane` — no
realistic crash path, but folding it into `render_figures` removes the coupling.

## Priority

1. **state_discrimination** — the only genuine crash-on-empty-fit.
2. **charge_gate_ramsey, ac_stark_shift, readout_pulse_photon, zz_interaction** — a
   high-value raw 2D map/trace coupled to a fit sibling. (`parametric_drive_decoherence`
   was in this tier and is DONE.)
3. **readout_fidelity** + **qubit_tomography** (mass-drop).
4. **ramsey, qubit_decoherence** — structural only; migrate for consistency.

## Fix pattern (all cases)

```python
return render_figures({
    "raw_2d": lambda: plot_raw(plot_data),
    "shift_vs_power": lambda: plot_shift(plot_data),
}, label=self.estimator_name)
```
Plus, for `state_discrimination`, guard the residue plotter (`np.nanmin/nanmax` on an
all-NaN array → finite fallback, or skip the residue colorbar when the fit failed), and
add a per-estimator "figures render on a failed fit" test. Reference implementation:
`spectroscopy_cryoscope/estimator.py` + `visualization.py`.
