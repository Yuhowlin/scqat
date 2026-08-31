from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import cos, abs, max, min, mean, argmax, pi
from numpy import fft
import numpy as np

from .function_fitting import FunctionFitting, register_fitter, parse_xy


@register_fitter('cosine')
class FitCosine(FunctionFitting):
    """
    Fit a (non-decaying) cosine model:
        a * cos(2*pi*f*x + phi) + c

    Accepts an ``xarray.DataArray`` with an ``'x'`` coordinate, raw ``(x, y)``
    arrays, or a bare ``y`` array (``x`` then defaults to the sample index), via
    the shared :func:`parse_xy` helper.

    Ported from qcat ``fit_cosine`` — generalised to the flexible :func:`parse_xy`
    input and the scqat ``fit()`` convention (``guess()`` runs only when
    parameters have not already been set, so a caller may tweak the guess).
    """

    def __init__(self, data: DataArray = None, x=None):
        self._data_parser(data, x)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, a, f, phi, c):
        return a * cos(2 * pi * f * x + phi) + c

    def guess(self):
        y = self.y
        t = self.x
        dt = float(t[1] - t[0])
        max_val = float(max(y))
        min_val = float(min(y))

        # FFT for the frequency guess
        amp = fft.fft(y)[: len(y) // 2]
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # remove DC
        power = abs(amp)
        f_guess = float(abs(freq[argmax(power)]))

        f_dict = dict(value=f_guess, min=0.0, max=1.0 / dt / 2)
        phi_dict = dict(value=0.0, min=-pi, max=pi)
        c_dict = dict(value=float(mean(y)), min=min_val, max=max_val)
        a_guess = (max_val - min_val) / 2
        a_dict = dict(value=a_guess, min=0.0, max=a_guess * 2 if a_guess > 0 else 1.0)

        self.params = self.model.make_params(a=a_dict, f=f_dict, phi=phi_dict, c=c_dict)
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result


def _frequency_seeds(x, y):
    """Candidate frequency seeds for a cosine fit: sub-bin refined, then raw.

    The FFT peak is only accurate to half a bin (``1/(N*dx)``), which is coarse
    for the short integer-N axes a swap sweep uses -- 21 points put the bins
    0.048 apart, so a true frequency can sit a quarter of a bin from the nearest
    one and pin the fit there. Interpolating a parabola through the peak bin and
    its two neighbours recovers the sub-bin position (the standard three-point
    estimator), which lands close enough for the optimiser to walk in.

    Returns the refined seed first and the raw bin second, so the caller keeps
    whichever actually fits better and never does worse than before.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 4 or x.size < 4:
        return (None,)
    dx = float(np.min(np.diff(x))) if x.size > 1 else 1.0
    if not np.isfinite(dx) or dx <= 0:
        return (None,)
    spectrum = np.abs(np.fft.rfft(np.where(finite, y, np.nanmean(y[finite]))))
    spectrum[0] = 0.0                       # remove DC
    if spectrum.size < 3:
        return (None,)
    k = int(np.argmax(spectrum))
    raw = k / (y.size * dx)
    if k == 0 or k >= spectrum.size - 1:
        return (raw,)
    left, peak, right = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    denom = left - 2.0 * peak + right
    if denom == 0:
        return (raw,)
    # the three-point parabolic peak offset, clamped to the bin it belongs to
    delta = float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
    refined = (k + delta) / (y.size * dx)
    return (refined, raw)


def fit_swap_oscillation(x, y) -> dict:
    """Fit ONE swap-oscillation trace and report the exchange angle per swap.

    A coherent (partial) swap exchanges population between the pair swap by swap,
    so the measured population follows a non-decaying cosine in the swap count N.
    Fitting ``a*cos(2*pi*f*N + phi) + c`` gives the oscillation frequency ``f`` in
    cycles per swap, and hence

        swap_period = 1 / f          swaps per full population cycle
        theta_rad   = pi * f         the exchange ANGLE applied by ONE swap

    (a full iSWAP is ``theta = pi/2`` => ``f = 0.5``, the Nyquist limit of an
    integer-N sweep; a sqrt-iSWAP is ``theta = pi/4`` => a 4-swap period.)

    This is the per-trace REDUCTION shared by every experiment that counts swaps:
    ``swap_oscillation`` fits a single trace with it, and ``pair_swap_angle``
    calls it once per row of a (knob x swap_count) map. It lives in ``tools/``
    because sharing a numerical routine is how two estimators share math without
    a second model binding.

    Parameters
    ----------
    x : array_like
        The swap counts N (integers; ``N=0`` — the no-swap baseline — is fine).
    y : array_like
        The measured signal at each N (a population, or a raw quadrature).

    Returns
    -------
    dict
        ``a``, ``f``, ``phi``, ``c`` (the cosine parameters), ``theta_rad``,
        ``swap_period``, ``r_squared``, ``success``, plus the arrays ``best_fit``
        (sampled at ``x``), ``x_dense`` / ``best_fit_dense`` (a 501-point curve
        for smooth drawing) and the lmfit ``fit_report``.

        **Never raises on bad data.** A trace that is too short, all-NaN, or
        unfittable degrades to NaN fields with ``success=False``, so a caller
        looping over map rows keeps its raw figure — the "raw data must always be
        plottable" rule.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def _degenerate(reason: str) -> dict:
        dense = (np.linspace(float(np.min(x)), float(np.max(x)), 501)
                 if x.size > 1 else np.full(501, np.nan))
        return {
            "a": float("nan"), "f": float("nan"), "phi": float("nan"),
            "c": float("nan"), "theta_rad": float("nan"),
            "swap_period": float("nan"), "r_squared": float("nan"),
            "success": False,
            "best_fit": np.full(x.shape, np.nan),
            "x_dense": dense,
            "best_fit_dense": np.full(501, np.nan),
            "fit_report": reason,
        }

    # The cosine has four free parameters, so fewer than four finite points
    # cannot constrain it; an all-NaN row is a failed acquisition.
    finite = np.isfinite(y)
    if x.size < 4 or finite.sum() < 4:
        return _degenerate("too few finite points to fit a cosine")

    # Two seeded parameters, each with its own trap.
    #
    # PHASE: FitCosine bounds the amplitude a >= 0, so a single phi=0 seed can
    # get trapped at a flat (a~0) fit for a trace whose population RISES from
    # zero (the swap target, which needs phi ~ pi).
    #
    # FREQUENCY: guess() seeds from the FFT peak BIN, and the bins are spaced
    # 1/(N*dx). A true frequency landing between two bins pins the fit at the
    # nearer bin and it never walks off -- silently, with a plausible angle and a
    # collapsed R^2. That is not a rare corner: nothing makes a swap period fall
    # on a bin, and it cost one rejected row per handful in the offline sweep.
    # So the bin is REFINED to sub-bin precision first (see _refined_frequency)
    # and the raw bin is kept as a second candidate.
    #
    # Fit every (frequency, phase) seed and keep the lowest-residual result.
    fit_result = None
    try:
        for f_seed in _frequency_seeds(x, y):
            for phi_seed in (0.0, pi):
                fitter = FitCosine(y, x)
                fitter.guess()
                fitter.params["phi"].set(value=phi_seed)
                if f_seed is not None:
                    lo = fitter.params["f"].min
                    hi = fitter.params["f"].max
                    fitter.params["f"].set(value=float(np.clip(f_seed, lo, hi)))
                res = fitter.fit()
                if fit_result is None or res.chisqr < fit_result.chisqr:
                    fit_result = res
    except Exception as err:  # noqa: BLE001 - a bad row must not sink the map
        return _degenerate(f"{type(err).__name__}: {err}")

    p = {k: v.value for k, v in fit_result.params.items()}  # a, f, phi, c

    nyquist = 0.5 / float(np.min(np.diff(x))) if x.size > 1 else float("nan")
    swap_period = 1.0 / p["f"] if p["f"] > 0 else float("nan")

    # Reject a degenerate (flat) fit: a real swap oscillation has a ~ half the
    # signal peak-to-peak, so guard against a ~ 0 falsely reporting success.
    ptp = float(np.ptp(y[finite]))
    contrast_ok = ptp > 0 and p["a"] > 0.05 * ptp

    # The contrast guard is relative to the signal's own peak-to-peak, so a
    # pure-noise (no-op swap macro) curve passes it trivially. Require the cosine
    # to also beat a constant model: R^2 of the fit vs the mean. Noise-fitting
    # gives R^2 ~ 0.2; a real oscillation gives ~ 0.99 - 0.5 separates cleanly
    # and is scale-free (works for population and raw-I signals alike).
    ss_tot = float(np.sum((y[finite] - y[finite].mean()) ** 2))
    r_squared = 1.0 - float(fit_result.chisqr) / ss_tot if ss_tot > 0 else float("nan")
    quality_ok = np.isfinite(r_squared) and r_squared > 0.5

    success = bool(
        bool(fit_result.success)
        and np.isfinite(nyquist)
        and 0 < p["f"] <= nyquist
        and contrast_ok
        and quality_ok
    )

    # Dense fit curve for plotting: the sweep has only a handful of integer-N
    # points, so the best-fit sampled there draws as a jagged polyline.
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 501)
    best_fit_dense = np.asarray(fit_result.eval(x=x_dense), dtype=float)

    return {
        "a": p["a"],
        "f": p["f"],
        "phi": p["phi"],
        "c": p["c"],
        # The angle ONE swap applies. f is in cycles per swap and a full
        # population cycle is 2*theta of exchange, so theta = pi * f.
        "theta_rad": float(pi * p["f"]),
        "swap_period": float(swap_period),
        "r_squared": float(r_squared),
        "success": success,
        "best_fit": np.asarray(fit_result.best_fit, dtype=float),
        "x_dense": x_dense,
        "best_fit_dense": best_fit_dense,
        "fit_report": fit_result.fit_report(),
    }
