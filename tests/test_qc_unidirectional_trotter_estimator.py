"""Synthetic-chain tests for the unidirectional-Trotter transport estimator.

The dataset is the WHOLE three-qubit chain (no per-target split), so the tests
pin both projections: the per-qubit transport curves that are always present,
and the joint distribution that only a shot-mode run carries.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_unidirectional_trotter import QcUnidirectionalTrotterEstimator
from scqat.estimators.qc_unidirectional_trotter.estimator import (
    FIG_JOINT,
    FIG_POPULATIONS,
)

QUBITS = ["q1", "q2", "q3"]
ROLES = {"source": "q1", "relay": "q2", "sink": "q3"}
#: 3-qubit basis labels, digit order = QUBITS order.
LABELS = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]


def _chain_ds(n_max: int = 20, joint: bool = False) -> xr.Dataset:
    """A cascade: the source decays, the sink fills, the relay stays reset."""
    n = np.arange(0, n_max + 1)
    p_source = 0.95 * np.exp(-n / 5.0)
    p_sink = 0.85 * (1.0 - np.exp(-n / 5.0))
    p_relay = np.full(n.shape, 0.04)
    pop = np.stack([p_source, p_relay, p_sink])
    ds = xr.Dataset(
        {"population": (("qubit", "round_count"), pop)},
        coords={"qubit": QUBITS, "round_count": n},
    )
    if joint:
        # A single excitation living on the source or the sink, plus a small
        # ground-state remainder — normalized over the 8 basis states.
        probs = {"100": p_source, "001": p_sink}
        rows = [probs.get(label, np.zeros(n.shape)) for label in LABELS]
        rows[LABELS.index("000")] = np.clip(1.0 - p_source - p_sink, 0.0, 1.0)
        jp = np.stack(rows)
        jp = jp / jp.sum(axis=0, keepdims=True)
        ds["joint_population"] = (("joint_state", "round_count"), jp)
        ds = ds.assign_coords(joint_state=LABELS)
    return ds


def test_summarizes_each_chain_qubit():
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds()
    est._check_data(ds)
    res = est.extract_parameters(ds, **ROLES)

    assert res["qubits"] == QUBITS
    assert res["n_round_count"] == 21 and res["max_round_count"] == 20.0
    assert res["has_joint"] is False
    # the source starts excited and ends near empty; the sink does the reverse
    assert res["per_qubit"]["q1"]["p_initial"] == pytest.approx(0.95)
    assert res["per_qubit"]["q1"]["p_final"] < 0.05
    assert res["per_qubit"]["q3"]["p_initial"] == pytest.approx(0.0, abs=1e-9)
    assert res["per_qubit"]["q3"]["p_final"] > 0.8
    # the relay is emptied every round, so it never carries the excitation
    assert res["per_qubit"]["q2"]["p_max"] < 0.1
    # the transport headline is the sink's peak — what SCQO's min_transfer reads
    assert res["sink_p_max"] == pytest.approx(res["per_qubit"]["q3"]["p_max"])
    assert res["sink_n_at_max"] == 20.0


def test_axis_order_is_irrelevant():
    """Estimators transpose by coordinate NAME, so a caller may pass any order."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds()
    flipped = ds.transpose("round_count", "qubit")
    assert est.extract_parameters(flipped, **ROLES) == est.extract_parameters(ds, **ROLES)


def test_joint_is_summarized_and_plotted_only_when_present():
    est = QcUnidirectionalTrotterEstimator()

    marginal_only = est.build_plot_data(_chain_ds(), {}, **ROLES)
    assert "joint_population" not in marginal_only.data_vars
    assert marginal_only.attrs["has_joint"] == 0
    assert set(est.generate_figures(None, None, plot_data=marginal_only)) == {
        FIG_POPULATIONS
    }

    ds = _chain_ds(joint=True)
    est._check_data(ds)
    res = est.extract_parameters(ds, **ROLES)
    assert res["has_joint"] is True
    # the excitation is only ever on the source or the sink
    assert res["joint_p_max"]["100"] > 0.9
    assert res["joint_p_max"]["001"] > 0.8
    assert res["joint_p_max"]["111"] == pytest.approx(0.0, abs=1e-9)

    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert plot_data.attrs["has_joint"] == 1
    assert list(plot_data["joint_state"].values) == LABELS
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {
        FIG_POPULATIONS,
        FIG_JOINT,
    }


def test_plot_data_roundtrips_through_netcdf(tmp_path):
    """Artifacts must reload with zero re-fit — so no attr may be non-netCDF."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds(joint=True)
    plot_data = est.build_plot_data(ds, est.extract_parameters(ds, **ROLES), **ROLES)
    path = tmp_path / "plotdata.nc"
    plot_data.to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        assert reloaded.attrs["source"] == "q1"
        figs = est.generate_figures(None, None, plot_data=reloaded.load())
    assert set(figs) == {FIG_POPULATIONS, FIG_JOINT}


def test_figures_render_on_a_failed_acquisition():
    """An all-NaN chain must still produce BOTH figures, not zero of them."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds(joint=True)
    ds["population"] = ds["population"] * np.nan
    ds["joint_population"] = ds["joint_population"] * np.nan

    res = est.extract_parameters(ds, **ROLES)
    assert np.isnan(res["per_qubit"]["q1"]["p_max"])
    assert np.isnan(res["sink_p_max"])
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {
        FIG_POPULATIONS,
        FIG_JOINT,
    }


def test_single_round_point_still_draws():
    """max_rounds=0 leaves one column; the joint map must not fail on it."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds(n_max=0, joint=True)
    plot_data = est.build_plot_data(ds, est.extract_parameters(ds, **ROLES), **ROLES)
    assert set(est.generate_figures(ds, {}, plot_data=plot_data)) == {
        FIG_POPULATIONS,
        FIG_JOINT,
    }


@pytest.mark.parametrize(
    "mangle, message",
    [
        (lambda ds: ds.drop_vars("population"), "population"),
        (lambda ds: ds.rename({"round_count": "N"}), "round_count"),
        (lambda ds: ds.drop_vars("joint_state"), "joint_state"),
    ],
)
def test_check_data_names_what_is_missing(mangle, message):
    est = QcUnidirectionalTrotterEstimator()
    with pytest.raises(ValueError, match=message):
        est._check_data(mangle(_chain_ds(joint=True)))


def test_analyze_writes_the_artifacts(tmp_path):
    est = QcUnidirectionalTrotterEstimator()
    results, figures = est.analyze(
        _chain_ds(joint=True), output_dir=str(tmp_path), **ROLES
    )
    assert results["sink_p_max"] > 0.8
    assert set(figures) == {FIG_POPULATIONS, FIG_JOINT}
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_unidirectional_trotter_metadata.json" in written
    assert "qc_unidirectional_trotter_plotdata.nc" in written
    # the estimator-named key drops the prefix; the other one keeps it
    assert f"{FIG_POPULATIONS}.png" in written
    assert f"{FIG_POPULATIONS}_{FIG_JOINT}.png" in written
