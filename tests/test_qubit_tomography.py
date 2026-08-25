"""Qubit-tomography estimator over the flat ``tools.discriminate`` reduction.

First direct coverage for this estimator — including the degenerate-training
regression: ragged ``direct_counts`` from a shot-less GMM centre used to raise
``IndexError`` at ``counts[1, 1]``; the fixed-shape reduction must instead
yield a graceful ``success=False``.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators import QubitTomographyEstimator
from scqat.estimators.state_discrimination import StateDiscriminationEstimator


def _tomo_dataset(n_train=300, n_shot=200, sep=8.0, sigma=1.0, seed=0,
                  degenerate=False):
    """Minimal tomography dataset: z-basis only, one gate count, two training
    states. ``degenerate=True`` puts BOTH training states in the ground blob
    (e.g. the pi pulse was dead) so one GMM centre captures every shot."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [sep, 0.0]])

    I_train = np.empty((2, n_train))
    Q_train = np.empty((2, n_train))
    for s in range(2):
        c = centers[0] if degenerate else centers[s]
        I_train[s] = c[0] + sigma * rng.standard_normal(n_train)
        Q_train[s] = c[1] + sigma * rng.standard_normal(n_train)

    # Tomography shots: all in the excited blob (population_z -> 1).
    I_tomo = centers[1][0] + sigma * rng.standard_normal((1, 1, 1, n_shot))
    Q_tomo = centers[1][1] + sigma * rng.standard_normal((1, 1, 1, n_shot))

    return xr.Dataset(
        {
            "I_tomo": (("basis", "sym", "gate_count", "shot_idx"), I_tomo),
            "Q_tomo": (("basis", "sym", "gate_count", "shot_idx"), Q_tomo),
            "I_train": (("prepared_state", "train_shot_idx"), I_train),
            "Q_train": (("prepared_state", "train_shot_idx"), Q_train),
        },
        coords={
            "basis": ["z"], "sym": ["reg"], "gate_count": [0],
            "shot_idx": np.arange(n_shot), "prepared_state": [0, 1],
            "train_shot_idx": np.arange(n_train),
        },
    )


def test_healthy_training_classifies_and_succeeds():
    ds = _tomo_dataset()
    res = QubitTomographyEstimator().extract_parameters(ds)
    assert res["success"]
    assert res["readout_fidelity"] > 0.9
    assert np.asarray(res["confusion_matrix"]).shape == (2, 2)
    # All tomo shots were prepared in |1>.
    assert res["population_z"][0] == pytest.approx(1.0, abs=0.05)


def test_degenerate_training_fails_gracefully():
    """One GMM centre captures no shot: previously an IndexError at
    counts[1, 1]; now a well-formed unsuccessful result."""
    ds = _tomo_dataset(degenerate=True)
    res = QubitTomographyEstimator().extract_parameters(
        ds, user_mean=[[0.0, 0.0], [50.0, 50.0]], user_std=1.0
    )
    assert res["success"] is False
    assert res["readout_fidelity"] == pytest.approx(0.5, abs=0.05)
    assert np.asarray(res["confusion_matrix"]).shape == (2, 2)


def test_unknown_kwarg_raises():
    ds = _tomo_dataset(n_train=50, n_shot=20)
    with pytest.raises(ValueError, match="user_stdd"):
        QubitTomographyEstimator().extract_parameters(ds, user_stdd=1.0)


def test_state_discrimination_wrapper_contract():
    """The thin StateDiscriminationEstimator keeps its public result keys and
    its artifacts (metadata JSON-shape + plot data + 4 figures) intact."""
    ds = _tomo_dataset()
    sd_ds = xr.Dataset(
        {"I": ds["I_train"], "Q": ds["Q_train"]}
    ).rename({"train_shot_idx": "shot_idx"})

    est = StateDiscriminationEstimator()
    res = est.extract_parameters(sd_ds)
    assert set(res) == {
        "trained_paras", "fitted_paras", "gaussian_norms", "direct_counts",
        "state_label", "outlier_mask", "outlier_probability", "norm_res",
        "fit_residues", "hist_dataset",
    }
    assert res["direct_counts"].shape == (2, 2)
    assert res["fit_residues"].dims == ("prepared_state", "y", "x")

    meta = est.extract_metadata(res)
    assert set(meta) == {
        "trained_paras", "fitted_paras", "gaussian_norms",
        "direct_counts", "outlier_probability", "norm_res",
    }
    pd = est.build_plot_data(sd_ds, res)
    assert set(est.generate_figures(sd_ds, res, plot_data=pd)) == {
        "raw", "2DHist", "outliers", "fit_residue",
    }


def test_interleaved_noise_tomography():
    """Interleaved noise dataset with noise_condition=['off', 'on']."""
    n_train = 100
    n_shot = 50
    sep = 8.0
    sigma = 0.5
    rng = np.random.default_rng(42)

    centers = np.array([[0.0, 0.0], [sep, 0.0]])
    I_train = np.empty((2, n_train))
    Q_train = np.empty((2, n_train))
    for s in range(2):
        I_train[s] = centers[s][0] + sigma * rng.standard_normal(n_train)
        Q_train[s] = centers[s][1] + sigma * rng.standard_normal(n_train)

    # I_tomo: shape (noise_condition=2, basis=3, sym=2, gate_count=3, shot_idx=50)
    # off: population 0 (closer to center 0); on: population 1 (closer to center 1)
    I_tomo = np.zeros((2, 3, 2, 3, n_shot))
    Q_tomo = np.zeros((2, 3, 2, 3, n_shot))

    # The `inv` slice must carry the INVERTED outcome, as the probe plays it: the
    # estimator symmetrises each condition as (p_reg + (1 - p_inv)) / 2, which maps
    # a UNIFORM population to exactly 0.5 whatever its value. Filling reg and inv
    # alike therefore reads 0.5 for both conditions, and the differential drift this
    # test exists to measure comes out identically zero.
    # Noise OFF: reg near center 0, inv near center 1  -> symmetrises to 0
    I_tomo[0, :, 0] = centers[0][0] + sigma * rng.standard_normal((3, 3, n_shot))
    I_tomo[0, :, 1] = centers[1][0] + sigma * rng.standard_normal((3, 3, n_shot))
    # Noise ON:  reg near center 1, inv near center 0  -> symmetrises to 1
    I_tomo[1, :, 0] = centers[1][0] + sigma * rng.standard_normal((3, 3, n_shot))
    I_tomo[1, :, 1] = centers[0][0] + sigma * rng.standard_normal((3, 3, n_shot))

    ds = xr.Dataset(
        {
            "I_tomo": (("noise_condition", "basis", "sym", "gate_count", "shot_idx"), I_tomo),
            "Q_tomo": (("noise_condition", "basis", "sym", "gate_count", "shot_idx"), Q_tomo),
            "I_train": (("prepared_state", "train_shot_idx"), I_train),
            "Q_train": (("prepared_state", "train_shot_idx"), Q_train),
        },
        coords={
            "noise_condition": ["off", "on"],
            "basis": ["x", "y", "z"],
            "sym": ["reg", "inv"],
            "gate_count": [0, 1, 2],
            "shot_idx": np.arange(n_shot),
            "prepared_state": [0, 1],
            "train_shot_idx": np.arange(n_train),
        },
    )

    est = QubitTomographyEstimator()
    res = est.extract_parameters(ds)

    assert res["success"] is True
    assert res["interleaved_noise"] is True
    assert len(res["differential_drift"]) == 3
    # Off was at 0, on was at 1, so delta population should be ~1.0
    assert np.all(np.array(res["differential_drift"]) > 0.5)

    plot_data = est.build_plot_data(ds, res)
    assert plot_data is not None
    assert "differential_drift" in plot_data
    assert "baseline_population_x" in plot_data

    figs = est.generate_figures(ds, res, plot_data=plot_data)
    assert "qubit_tomography_2d" in figs
    assert "qubit_tomography_3d" in figs
    assert "qubit_tomography_dist" in figs
    assert "qubit_tomography_diff" in figs
