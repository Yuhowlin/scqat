"""Run ``ParametricDriveDecoherenceEstimator`` on existing EP parametric-drive data.

This mirrors the per-frequency analysis in ``notebooks/EP/view_single_raw.ipynb``,
but goes through the packaged scqat estimator (the same one the renewed
``LCH_qubit_parametric_drive_freq_time`` / ``..._freq_time_tomo`` nodes call). For
each ``driving_frequency`` it reconstructs rho_11(t), fits the non-Markovian
amplitude-damping model, and reports gamma / lambda / Delta plus the
exceptional-point figure of merit ``8*lambda^2/gamma^2``.

Two real datasets are wired in as examples and auto-detected:

* ``rho11_only``  — freq_time node output (no tomography) -> rho_11-only path.
* ``tomography``  — freq_time_tomo node output (basis = X/Y/Z) -> full density matrix.

Usage
-----
Run every example dataset that exists on disk::

    python examples/parametric_drive/run_decoherence_estimator.py

Run one specific file (e.g. another acquisition)::

    python examples/parametric_drive/run_decoherence_estimator.py path/to/ds_raw.h5

Each run writes, per qubit, under ``examples/parametric_drive/output/<tag>/<qubit>/``:
``parametric_drive_decoherence_metadata.json``, ``..._plotdata.nc``, and the two
figures ``..._decoherence_params.png`` and ``..._rho11_fits.png``.
"""

from __future__ import annotations

import os
import sys

# Make the repo importable when run as a plain script (no install required).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt

from scqat.parsers.xarray_h5_parser import load_xarray_h5
from scqat.parsers import repetition_data
from scqat.estimators import ParametricDriveDecoherenceEstimator

# --- Example data on disk -----------------------------------------------------
# These are the run folders this example was written against. Set
# SCQAT_EXAMPLE_DATA_ROOT to the directory holding them, or just pass your own
# ds_raw.h5 on the command line (see the README). With the variable unset the
# paths stay relative and simply will not be found, which the script reports.
_DATA_ROOT = os.environ.get("SCQAT_EXAMPLE_DATA_ROOT", "")
EXAMPLE_DATA = {
    "rho11_only": os.path.join(
        _DATA_ROOT,
        "QtoR/20260202/#1354_LCH_qubit_parametric_drive_time_13_034031/ds_raw.h5",
    ),
    "tomography": os.path.join(
        _DATA_ROOT,
        "QtoR/20260210_tomo/x180/"
        "#1518_LCH_qubit_parametric_drive_time_tomo_10_001849/ds_raw.h5",
    ),
}
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Readout-contrast correction applied to the Z-basis population before fitting.
# These default to the estimator's built-in values; override per dataset here if
# your readout zero / contrast differ.
ESTIMATOR_KWARGS: dict = {
    # "rho11_offset": 0.045,
    # "rho11_scale": 0.78,
}


def run(h5_path: str, tag: str | None = None, *, show: bool = False, **estimator_kwargs):
    """Load one ds_raw.h5, run the estimator per qubit, and save the artifacts.

    Returns the output directory for this dataset.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    tag = tag or os.path.basename(os.path.dirname(h5_path))
    out_dir = os.path.join(OUTPUT_ROOT, tag)

    ds = load_xarray_h5(h5_path)
    per_qubit = repetition_data(ds, "qubit") if "qubit" in ds.dims else [ds]

    estimator = ParametricDriveDecoherenceEstimator()
    kwargs = {**ESTIMATOR_KWARGS, **estimator_kwargs}
    for sq in per_qubit:
        qname = str(sq["qubit"].values.item()) if "qubit" in sq.coords else "q"
        qdir = os.path.join(out_dir, qname)
        results, figs = estimator.analyze(sq, output_dir=qdir, verbose=False, **kwargs)
        layout = "tomography" if results["has_tomography"] else "rho_11-only"
        print(
            f"[{tag}/{qname}] {layout}: "
            f"freqs={results['n_freq']}  decoh_ok={results['n_decoh_ok']}  -> {qdir}"
        )
        if show:
            plt.show()
        else:
            plt.close("all")
    return out_dir


def main(argv):
    if len(argv) > 1:
        # A single explicit ds_raw.h5 path was given.
        run(argv[1], show=True)
    else:
        for tag, path in EXAMPLE_DATA.items():
            if os.path.exists(path):
                run(path, tag=tag)
            else:
                print(f"[skip] {tag}: file not found -> {path}")
    print(f"\nArtifacts (metadata + plot data + figures) under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main(sys.argv)
