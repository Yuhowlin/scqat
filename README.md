# scqat — superconducting-qubit analysis tool

`scqat` fits analysis models to superconducting-qubit data — from **experiment or
simulation** — and extracts physical parameters. It is a pip-installable library meant to be
imported by simulation/experiment repos; estimators are blind to where the data came from and
interact only with `xarray.Dataset`.

## Architecture

A decoupled, one-directional layering with `xarray.Dataset` as the universal data container:

```
workflows  →  estimators  →  tools
parsers    →  (nothing in scqat)
core       →  BaseEstimator ABC + I/O helpers
```

- **`parsers/`** — read raw files (HDF5, …) into an `xarray.Dataset`. No analysis.
  `from scqat.parsers import load_xarray_h5, repetition_data, parse_timestamp`
- **`estimators/`** — one `BaseEstimator` subclass per experiment, and one experiment per
  subclass (Ramsey, state discrimination, …). Output **metadata** (JSON) + **plot data**
  (netCDF).
  `from scqat.estimators import RamseyEstimator`
- **`tools/`** — shared, pure fitters/algorithms, discoverable via `get_fitter('<name>')`.
- **`workflows/`** — multi-estimator orchestration pipelines.

Each estimator returns two artifacts so a different (possibly non-Python) consumer can redraw
figures with no recomputation: `extract_parameters()` → metadata, `build_plot_data()` → a
self-sufficient plot-data `Dataset`, and `generate_figures()` draws from **plot data only**.

## Quickstart

```python
from scqat.parsers import load_xarray_h5, repetition_data
from scqat.estimators import RamseyEstimator

ds = load_xarray_h5("ds_raw.h5")
for sq in repetition_data(ds, repetition_dim="qubit"):
    metadata, figs = RamseyEstimator().analyze(sq, output_dir="out/")
```

## Docs

- Architecture & contribution rules: [`CLAUDE.md`](CLAUDE.md)
- QCAT→scqat migration status, feature backlog, and porting recipe: [`MIGRATION.md`](MIGRATION.md)

## Install

```bash
pip install -e .          # runtime
pip install -e ".[dev]"   # + pytest/jupyter
```

Requires Python ≥ 3.9.
