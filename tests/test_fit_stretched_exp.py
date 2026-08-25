"""Tests for the stretched (Kohlrausch) exponential fitter.

``a * exp(-(x/tau)**p) + c``. The exponent ``p`` names the noise that dominates
the dephasing (1 = Markovian, 2 = 1/f-dominated Gaussian) and is only measurable
over a wide dynamic range, which is why every test here uses a LOG-spaced axis.
"""

import numpy as np
import pytest

from scqat.tools.fit_stretched_exp import FitStretchedExponential, stretched_exp
from scqat.tools.function_fitting import get_fitter


def _log_axis(lo=1e-8, hi=2e-4, n=60):
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _decay(x, a=1.0, tau=2e-5, p=1.5, c=0.0, noise=0.0, seed=0):
    y = stretched_exp(x, a, tau, p, c)
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, x.size)
    return y


class TestFitStretchedExponential:
    def test_registered_under_its_name(self):
        x = _log_axis()
        fitter = get_fitter("stretched_exp", x=x, data=_decay(x))
        assert isinstance(fitter, FitStretchedExponential)

    @pytest.mark.parametrize("p", [0.8, 1.0, 1.5, 2.0])
    def test_recovers_all_four_parameters(self, p):
        x = _log_axis()
        result = FitStretchedExponential(x=x, data=_decay(x, p=p, noise=0.005)).fit()
        assert result.params["tau"].value == pytest.approx(2e-5, rel=0.1)
        assert result.params["p"].value == pytest.approx(p, rel=0.15)
        assert result.params["a"].value == pytest.approx(1.0, rel=0.1)
        assert result.params["c"].value == pytest.approx(0.0, abs=0.05)

    def test_fix_p_freezes_the_exponent(self):
        x = _log_axis()
        result = FitStretchedExponential(
            x=x, data=_decay(x, p=1.0, noise=0.005), fix_p=1.0
        ).fit()
        assert result.params["p"].value == pytest.approx(1.0)
        assert result.params["p"].vary is False
        assert result.params["tau"].value == pytest.approx(2e-5, rel=0.1)

    def test_models_the_rician_floor(self):
        """The consumer feeds a lock-in MAGNITUDE, which plateaus at the noise
        level instead of vanishing; left unmodelled that floor drags tau long."""
        x = _log_axis()
        floor = 0.08
        y = _decay(x, c=floor, noise=0.002)
        fitted = FitStretchedExponential(x=x, data=y).fit()
        assert fitted.params["c"].value == pytest.approx(floor, abs=0.02)
        assert fitted.params["tau"].value == pytest.approx(2e-5, rel=0.15)

        # pin the failure mode the offset exists to prevent
        forced = FitStretchedExponential(x=x, data=y)
        forced.guess()
        forced.params["c"].set(value=0.0, vary=False)
        assert forced.fit().params["tau"].value > fitted.params["tau"].value

    def test_tau_seed_is_scale_free_over_decades(self):
        """A span/2 seed (what FitExponentialDecay uses) is orders of magnitude
        off on a log axis; the 1/e crossing is not."""
        x = _log_axis()
        for tau in (1e-7, 1e-6, 2e-5):
            seed = FitStretchedExponential._tau_seed(x, _decay(x, tau=tau))
            assert seed == pytest.approx(tau, rel=2.0)

    def test_undecayed_record_still_fits_without_raising(self):
        """tau >> span must stay fittable and finite rather than blowing up.

        It does NOT stay meaningful: a record that never decays is degenerate
        (a -> 0 with c -> the plateau fits as well as tau -> infinity), so the
        returned tau is arbitrary. Catching that is the ESTIMATOR's job -- its
        rail and variance-explained gates reject exactly this shape -- not the
        fitter's, which is why nothing is asserted about the value here.
        """
        x = _log_axis()
        result = FitStretchedExponential(x=x, data=_decay(x, tau=1.0, noise=0.001)).fit()
        assert np.isfinite(result.params["tau"].value)
        assert np.isfinite(result.best_fit).all()

    def test_accepts_dataarray_and_raw_arrays_alike(self):
        from xarray import DataArray

        x = _log_axis()
        y = _decay(x, noise=0.002)
        from_arrays = FitStretchedExponential(x=x, data=y).fit()
        from_dataarray = FitStretchedExponential(
            data=DataArray(y, coords={"x": x}, dims=["x"])
        ).fit()
        assert from_dataarray.params["tau"].value == pytest.approx(
            from_arrays.params["tau"].value, rel=1e-6
        )

    def test_model_is_real_for_negative_excursions(self):
        """(x/tau)**p is real only for x >= 0; the clip keeps an lmfit excursion
        into a negative tau from returning NaN and killing the fit."""
        assert np.isfinite(stretched_exp(np.array([-1.0, 0.0, 1.0]), 1.0, -2.0, 1.5, 0.0)).all()
