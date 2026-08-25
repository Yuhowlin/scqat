# AI Assistant Guidelines: scqat

This repository (`scqat`) analyzes superconducting-qubit data. The data may come
from **either an experiment or a simulation** — estimators must be blind to which.
It strictly follows a decoupled, Domain-Driven architecture using
`xarray.Dataset` as the universal data transfer object.

> Cross-repo terminology (Experiment = probe + **estimator**) is defined in
> [SCQO](https://github.com/shiau109/SCQO)'s `CLAUDE.md` → Terminology. In short: an **Experiment**
> is a probe (acquisition) bound to an **estimator** (analysis). This repo implements the estimator
> + tool/fitter half. SCQO is a CONSUMER of scqat, never a dependency of it - scqat must stay
> importable on its own.

## Purpose & Consumption
`scqat` is a **pip-installable library** (`pyproject.toml`) meant to be imported
by other repositories:
- A **simulation** or **experiment** repo can `from scqat.tools import ...`
  to reuse the shared algorithms, or `from scqat.parsers import ...` to load a
  file it produced into the universal `xarray.Dataset`.
- Those repos hand a `Dataset` (or a file path) to an estimator in
  `scqat.estimators` and get back metadata + plot-reconstruction data.

So every layer must stay importable in isolation and free of side effects at
import time.

## Core Architectural Rules
1. **Universal Data Container:** All data passed from parsers to estimators MUST
   be an `xarray.Dataset`. Raw data arrays become `DataArray` variables, sweep
   parameters become `Coordinates`, and instrument/simulation settings become
   `Attributes` (`.attrs`).
2. **Strict Separation of Concerns:**
   - `scqat/parsers/`: ONLY for reading raw files (HDF5, CSV, API payloads) from
     a given path — whether produced by an experiment or a simulation — and
     converting them to `xarray.Dataset`. NEVER put physics analysis or fitting
     logic here.
   - `scqat/estimators/`: ONE estimator per experiment (e.g. T1, Ramsey,
     MIST). They accept an `xarray.Dataset`, process it, and output derived
     metadata and figures. NEVER put file I/O or raw data loading here.
   - `scqat/tools/`: Shared, **pure** mathematical algorithms (fitting,
     FFTs, Hankel analysis, analytical solvers). Anything used by more than one
     estimator lives here.
   - `scqat/workflows/`: Multi-estimator **orchestration pipelines** that chain
     parsers → several estimators/tools for a higher-level analysis (e.g.
     `ep_pipeline.py`). Pipelines return plain data structures; plotting is left
     to the caller. NEVER put file I/O of raw inputs anywhere except via parsers.
   - `scqat/core/base_estimator.py`: Defines the `BaseEstimator` ABC and the
     saving/loading helpers. The `analyze()` orchestrator optionally invokes
     these when `output_dir` is provided. Estimators MUST NOT perform file I/O
     outside this mechanism.
3. **Dependency direction (keeps the core reusable):** The import arrow points
   **one way only**: `workflows → estimators → tools`, and
   `parsers → (nothing in scqat)`. In particular `tools` MUST NOT import
   from `estimators`, `parsers`, or `workflows`. This is what lets an external
   simulation repo reuse `tools` without dragging in experiment logic.
4. **Format / provenance agnosticism:** Estimators must be completely blind to
   whether data came from simulation or experiment. They interact only with the
   `xarray.Dataset` API.

## Workflow Rules
1. **Plan before implementation.** When asked to generate code, explain the plan first.
   *In the maintainer's tree*, wait for the maintainer's approval before modifying existing
   code — that checkout is shared and live. Working in your own fork, "approval" is the pull
   request: propose on a branch, and let review be the gate. Either way the plan comes first,
   because the architectural rules above (the import arrow, the output contract) are easier to
   check in a plan than in a diff.

## Release checklist
**After committing a release-worthy feature** (maintainers, before any release): write ONE
fragment file `RELEASES.d/<feature-slug>.toml` **in the SCQO repo** (format + rules in that
directory's `README.md`). Contributors working from a fork cannot do this and are not expected
to - draft the fragment in your pull request body instead, and the maintainer commits it once
the last repo's PR has landed — especially the scqat-floor coupling when SCQO
lazy-imports something new (silent-failure kind). One file per feature, written as
the feature's LAST step; the release agent consumes the fragments when cutting the
combo.

Cutting a release = **two steps, in this order**:
1. Bump `version` in `pyproject.toml` to `X.Y.Z` and commit.
2. Tag that commit `vX.Y.Z`.

The digit choice follows SCQO `RELEASING.md`'s version rule, applied to scqat's
own line (0.x: any breaking/additive fragment → y+1, fix-only → z+1).

The pyproject version MUST equal the tag: dependents (SCQO) declare real floors
like `scqat>=0.20.0`, resolved from package metadata (`importlib.metadata`), so a
tag whose tree still carries the old version breaks every downstream install.
Never retag or rewrite an existing tag — if a tagged tree has the wrong version,
cut the next number.

## Estimator Output Contract
**One compute, two projections:** `extract_parameters()` runs the heavy work
**once** and returns the rich `results` — the single source of truth.
`extract_metadata()` and `build_plot_data()` are pure projections of it (key
scalars vs. plot arrays), so nothing is recomputed. Estimators transpose the
input dataset **by coordinate name** (order-invariant), so callers may pass sweep
axes in any order.

From that, an estimator produces **one mandatory artifact and two optional ones**:

1. **Metadata (mandatory)** — the *key physical parameters* (e.g. `T1`,
   `frequency`, `fwhm`), small and JSON-serializable, persisted as
   `<estimator_name>_metadata.json` (stamped with `estimator_name`).
   `extract_metadata(results)` projects the subset to keep; it defaults to the
   identity (a simple estimator's `results` *is* the metadata) and is overridden
   only to drop bulky intermediates. The parameter set may evolve (rarely), and
   JSON's flexible schema absorbs that.
2. **Plot data (optional)** — the *minimal arrays needed to redraw every figure
   with zero recalculation* (only trivial unit conversion, e.g. Hz→MHz, is
   allowed downstream). Returned by `build_plot_data()` as a single
   `xarray.Dataset` and saved as `<estimator_name>_plotdata.nc` (netCDF). Default
   is `None` (no plot-data artifact).
3. **Figures (optional)** — returned by `generate_figures()`. Default is `{}`.
   Because figures draw only from plot data, **providing figures implies
   providing plot data.**

The metadata/plot-data split exists so a *different* repo (possibly not even
Python) can reload the plot data and reconstruct the figures without rerunning
any analysis and without unpickling. JSON + netCDF are both self-describing and
language-agnostic — never use `pickle` for these artifacts.

**Self-enforcing rule:** `generate_figures()` must draw using **only** the
`plot_data` Dataset, never the raw input `dataset` or the working `results`. If a
figure can be drawn from `plot_data` alone, an external consumer can too —
anything a figure needs therefore has to be put into `build_plot_data()`.
(During migration the orchestrator still passes `dataset` and `results` to
`generate_figures` so older estimators keep working; new/migrated estimators must
ignore them.)

## Raw data must always be plottable

A run whose FIT failed must still produce its raw-data figure. A broken fit — or
a plotter that crashes on a degenerate (all-NaN) fit — must never leave the run
figure-less: SCQO's artifact fallback silently drops **all** figures on any single
plotter exception, so one crash = zero PNGs (that is exactly how a real
`qubit_spectroscopy_cryoscope` run lost its raw spectrogram to a log-scale crash
in the empty step-response panel). Four rules:

1. `build_plot_data` ALWAYS carries the raw measured arrays and never raises on a
   failed fit — fit-derived fields degrade to NaN, never absent.
2. `generate_figures` returns `render_figures({name: lambda: plot_x(plot_data),
   ...}, label=self.estimator_name)` (`scqat/core/figures.py`) — per-figure
   isolation, so one figure's failure is skipped with a warning and never drops
   another (crucially, the raw one). Pass THUNKS, not pre-built figures.
3. Each plotter draws its raw data UNCONDITIONALLY and guards the fit overlay: no
   `set_xscale("log")` + `tight_layout()` on an all-NaN series, no `set_*lim` /
   annotation that assumes a finite fit. Reference guard: the cryoscope
   `plot_step_response`.
4. A pure FIT view (no raw data) is exempt from rule 3's "raw" clause but still
   must not crash — it renders an annotated-empty panel.

Enforced per estimator by a "figures render on a failed fit" test (reference:
`tests/test_spectroscopy_cryoscope_estimator.py::...test_figures_render_on_a_failed_fit`).

## Multi-method estimators (N approaches, one physics)

When more than one analysis approach can extract the same physical parameters
(reference implementation: `estimators/resonator_spectroscopy/` — `lorentzian`
joint-background fit vs `circle` Probst notch fit), structure it as follows:

1. **Still ONE estimator per experiment.** Approaches are *method strategy
   objects* in a `methods/` subpackage (`methods/base.py` ABC + one module per
   method + a `METHODS` registry in `methods/__init__.py`); heavy math stays in
   `tools/` fitters. Selection is a plain `method=` kwarg on
   `extract_parameters` (default = the cheap/robust method).
2. **Two-tier result contract.** The estimator declares `COMMON_KEYS` (same
   name ⇒ same meaning AND unit in every method) and validates them right after
   `extract()`; orchestration (SCQO) may rely only on those. Everything else is
   method-owned extras, consumed downstream only via `if key in results`. Never
   reuse a key name across methods with a different meaning; error keys are
   common-but-best-effort (NaN allowed).
3. **Provenance.** `results["method"]` (→ metadata JSON) and
   `plot_data.attrs["method"]` are always stamped.
4. **Plots are method-dependent; artifacts are not.** Each method owns its
   plot_data variables and figure layout, but the figure dict key (and hence the
   PNG name) is identical for all methods, plot_data stays netCDF-safe (no
   complex variables — store I/Q float pairs), and `generate_figures` dispatches
   on `plot_data.attrs["method"]` — never on estimator state — so a saved
   `plotdata.nc` replots with zero re-fit.
5. **Adding method #N** = one module in `methods/` + one registry entry (+ one
   Literal value in the SCQO experiment's Parameters). Nothing else moves.
6. **Cross-method test.** The estimator's test suite includes an agreement test:
   same synthetic data → same COMMON physics within tolerance (a method changes
   robustness, not physics).

## Layered analyses (an experiment that contains another experiment's fit)

**Estimators never call estimators.** When an analysis needs another
experiment's fit as an inner step (the vs-flux/vs-power maps fit a resonator
dip per slice), the shared piece is a *pure per-trace reduction* — and by rule
it lives in `tools/` (references: `tools/dip_fit.py`, `fit_dip()` +
`DIP_METHODS` + `DIP_KNOBS`, for the resonator family; `tools/peak_fit.py`,
`fit_peaks()` + `PEAK_KNOBS`, plus the generic 2-D tracker
`tools/peak_map.py::track_peaks` for the qubit family), consumed by every
estimator in the family. When two sweep experiments share the whole map
reduction (vs-flux and parametric-drive), the tracker is generic-keyed
(`x`/`y`) in `tools/` and each estimator relabels into its own vocabulary; an
experiment's *Dataset-shaped* stage helpers live in its own subpackage
(reference: `resonator_spectroscopy_flux/dips.py`,
`qubit_spectroscopy_flux/peaks.py`) and may be imported as plain functions by
a downstream composite estimator of the same family or by control repos.
Estimator→estimator calls conflate the experiment-level contract
(metadata/plot_data/figures) with plain math, and their flat `**kwargs`
namespace makes inner options unreachable or silently mis-routed.

1. **"Same experiment + extra sweep axis" shares the reduction, not the
   estimator.** Only the per-trace fit is common; how it is driven (candidate
   seeding from the 2-D map, local windows, fallbacks), the cross-axis
   acceptance gates, the second-stage model, and the artifacts are all
   sweep-specific and belong to the swept experiment's own estimator.
2. **Flat, fully-owned kwarg surface.** The estimator's primary method axis is
   `method`; a secondary axis is `<thing>_method` (e.g. `dip_method`); every
   kwarg is documented in `extract_parameters`'s docstring. No prefixes, no
   forwarding of "whatever is left over".
3. **Validate before loops.** tools expose their valid-knob sets
   (`validate_dip_kwargs`); callers validate ONCE before any per-slice loop, so
   a typo'd kwarg raises instead of being swallowed by per-slice `try/except`
   fallbacks.
4. **Required keys only.** A caller of a tools reduction may rely only on its
   documented required keys (`detuning`/`fwhm`/`success` for `fit_dip`);
   method-owned extras (e.g. `amplitude`) via `.get` with defined degraded
   behavior.
5. **Provenance.** Every resolved `*_method` value is stamped in results, the
   metadata JSON, and `plot_data.attrs` (strings; bools as int 0/1; absent
   axes omitted). SCQO mirrors each axis as a `Literal` kept equal to the
   registry by a sync test (SCQO `tests/test_estimator_method_sync.py`).

## Implementation Guide

### Adding a new data format
Create a new script in `parsers/`. Write a function that takes a file path or API
payload (from experiment or simulation) and returns an `xarray.Dataset`.

### Adding a new estimator
Create a new class in `estimators/` that inherits from `BaseEstimator` (found in
`scqat/core/base_estimator.py`).

Every subclass MUST:
1. Set the class attribute `estimator_name` (str) — controls default output
   filenames.
2. Implement `extract_parameters(dataset, **kwargs) -> Dict[str, Any]` — the
   heavy compute returning the full `results`. This is the **only** required
   method; for a simple estimator its return *is* the metadata.

Every subclass SHOULD also:
3. Override `_check_data(dataset)` to validate that all required coordinates and
   variables are present. Raise `ValueError` with a descriptive message on
   failure.
4. Document the **dataset contract** — the exact variable, coordinate, and
   attribute names the estimator expects — in the module or class docstring.

A subclass MAY (optional — each has a safe default; semantics per **Estimator
Output Contract** above):
5. Override `extract_metadata(results) -> Dict[str, Any]` (default: identity).
6. Override `build_plot_data(dataset, results, **kwargs) -> xr.Dataset`
   (default: `None`).
7. Override `generate_figures(dataset, results, plot_data=None, **kwargs)
   -> Dict[str, plt.Figure]` (default: `{}`; requires `build_plot_data`).

The inherited `analyze()` method orchestrates:
`_check_data` → `extract_parameters` → `extract_metadata` → `build_plot_data` →
(optional save of metadata + plot data) → `generate_figures` → (optional save of
figures).

- **Every estimator is a subpackage** — one uniform layout, so there is no
  per-estimator "should I fold this into a single module?" judgment call:
  ```
  estimators/<name>/
      __init__.py      # re-exports the estimator class
      estimator.py     # the BaseEstimator subclass
      visualization.py # estimator-specific plotting helpers (consume plot_data)
  ```
  `visualization.py` is present whenever the estimator draws figures (almost
  always); a pure-fit estimator with no figures omits it but stays a subpackage,
  so adding a plot later is a new file, not a module→package restructure. The
  `__init__.py` MUST re-export the estimator class so external code can always use
  `from scqat.estimators.<name> import <Estimator>`.

- **`estimators/__init__.py` aggregation:** Every new estimator MUST also be
  imported in `scqat/estimators/__init__.py` so all estimators are available via
  `from scqat.estimators import <Estimator>`.

### Adding a new fitter to `tools/`
All fitters inherit from `FunctionFitting` (in
`scqat/tools/function_fitting.py`) and must:
1. Decorate the class with `@register_fitter('<name>')` so it is discoverable via
   the `get_fitter('<name>')` factory.
2. Accept flexible input so external (simulation) callers can use it without
   wrapping: an `xarray.DataArray` with an `'x'` coordinate, **or** raw `(x, y)`
   arrays, **or** a bare `y` array. Use the shared `parse_xy` helper in
   `function_fitting.py` to normalize the input.
3. Implement `model_function`, `guess`, and `fit` methods.
4. Write `pytest` tests in `tests/` for the new fitter (estimator-level tests are
   optional).

### Testing discipline — run only what the edit can break
Command (from the repo root): `uv run --extra dev pytest tests/test_ramsey_estimator.py -q`.
**`--extra dev` is required** — pytest is an optional-dependency extra, so bare `uv run pytest`
dies with `Failed to spawn: pytest`. Tests are named after what they test, so selection is by
**file name**, not `-k`. The blast radius follows the import arrow (rule 3): an estimator edit is
local; a `tools/` edit reaches every family that imports it.

| Edited | Run |
|---|---|
| `estimators/<name>/` | `tests/test_<name>_estimator.py` (naming varies — also `test_qubit_tomography.py`, `test_resonator_spectroscopy_flux_composite.py`) |
| a `methods/` strategy or a `METHODS` registry | that estimator's test file (it holds the cross-method agreement test, **Multi-method** rule 6) **+ SCQO's `tests/test_estimator_method_sync.py`** — the mirrored Literal lives in the *other* repo and nothing else catches the drift |
| `tools/<x>.py` | `tests/test_<x>.py` **+ every consuming family's test file** (below) |
| estimator↔tool structure (new subpackage, moved import) | add `tests/test_no_estimator_layering.py` — the static capstone for "estimators never call estimators" |
| `core/base_estimator.py`, `tools/function_fitting.py`, `estimators/__init__.py` | **full suite** — the `analyze()` orchestrator, the `FunctionFitting` base of every fitter, and the aggregate import touch everything |

<!-- BEGIN generated: tool-consumers -->
**GENERATED** - refresh with `python scripts/update_docs.py`. A `tools/` edit reaches
every family listed beside it; run those families' test files too.

| tool | consuming estimator families |
|---|---|
| `ade_decay` | qubit_t1_ade |
| `allan` | qubit_t1_bayesian |
| `dip_finder` | broadband_resonator_spectroscopy |
| `dip_fit` | resonator_spectroscopy, resonator_spectroscopy_flux, resonator_spectroscopy_power |
| `discriminate` | parity_switch_continuous, parity_switch_discrete, qubit_tomography, readout_fidelity, state_discrimination |
| `fit_abscos` | charge_gate_ramsey |
| `fit_cosine` | power_rabi, swap_oscillation |
| `fit_damped_oscillation` | qubit_echo_flux, zz_interaction |
| `fit_exp_decay` | qubit_echo, qubit_echo_flux, qubit_relaxation, qubit_relaxation_flux, qubit_t1_bayesian |
| `fit_gaussian2d` | single_state_outlier |
| `fit_lorentzian_bg` | resonator_spectroscopy |
| `fit_notch_circle` | resonator_spectroscopy |
| `fit_powerlaw_base` | qubit_sqrb |
| `fit_qubit_decoherence` | qubit_decoherence |
| `fit_stretched_exp` | ramsey_phasor |
| `fit_transmon_freq_flux` | qubit_flux_arch |
| `fit_triangle` | xyz_delay |
| `iq_reduce` | power_rabi, qubit_deterministic_benchmarking, qubit_echo, qubit_relaxation, qubit_spectroscopy_flux, qubit_stark_phase_echo, ramsey, xyz_delay |
| `lockin` | ramsey_cryoscope, ramsey_phasor |
| `peak_fit` | ac_stark_shift, broadband_qubit_spectroscopy, parametric_drive_resonance, qubit_spectroscopy, qubit_spectroscopy_flux, readout_pulse_photon, spectroscopy_cryoscope |
| `peak_map` | parametric_drive_resonance, qubit_spectroscopy_flux |
| `ramsey_fit` | charge_gate_ramsey, ramsey |
| `robust` | resonator_spectroscopy_flux |
| `step_response_fit` | ramsey_cryoscope, spectroscopy_cryoscope |
| `telegraph_psd` | parity_switch_continuous, parity_switch_discrete |
| `timeseries_psd` | qubit_t1_bayesian |

No estimator imports these (shared machinery, workflow-only, or a fitter reached through
the `get_fitter()` factory): `fit_damping_beat`, `fit_lorentzian`, `fit_multi_damped_oscillation`, `flux_predistortion`, `function_fitting`, `ge_discriminator`, `hankel`.
<!-- END generated: tool-consumers -->

**Coverage gap — do not paper over it.** If your edit is in one of the estimators below, say
so plainly instead of reporting a targeted run as though it proved something. **This is
deliberate, not neglect:** these are not yet part of the scqo system, and each gets its tests
when it is promoted in — SCQO's promotion checklist already requires "`simulate()` implemented
→ offline end-to-end test in `tests/`". Do not open a campaign to backfill them.

<!-- BEGIN generated: coverage-gap -->
**GENERATED** - refresh with `python scripts/update_docs.py`. **10 of 43**
estimators are imported by NO test, by module path or exported class:

- `ac_stark_shift`
- `broadband_qubit_spectroscopy`
- `charge_gate_ramsey`
- `qubit_decoherence`
- `qubit_drag_alternating`
- `qubit_drag_equator`
- `qubit_sqrb`
- `readout_pulse_photon`
- `single_state_outlier`
- `zz_interaction`
<!-- END generated: coverage-gap -->

The **full suite** (`uv run --extra dev pytest -q`) is for cutting a release or
a shared-core edit (row 5). Otherwise **report the exact command run** and offer the full-suite
command rather than spending the time unasked.

## Offline analysis on saved data
Estimators can be iterated against **real saved runs** (`ds_raw.h5` / `plotdata_*.h5` written
by a driver) without re-running hardware. A saved `plotdata_*.h5` *is* the estimator-native
`build_plot_data` Dataset (see **Estimator Output Contract**): reload it with
`from scqat.parsers import load_xarray_h5` and draw via
`estimator.generate_figures(None, None, plot_data=…)` — no re-fit, no parsing. That is the
whole trick; a scratch `# %%`-cell script per experiment is enough scaffolding.

The maintainer's own exploration tree (`analysis/`, `notebooks/`, `temp/`) is **not published**
— it holds absolute paths into lab storage and is kept local. Nothing in `tests/` depends on it.

**Never put external or absolute data paths in `tests/`.** Tests use synthetic data or a small
committed fixture; path-based exploration belongs in your own scratch scripts, which is exactly
why that tree is unpublished.
