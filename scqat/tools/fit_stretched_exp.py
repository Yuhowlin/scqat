from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
import numpy as np

from .function_fitting import FunctionFitting, register_fitter, parse_xy


def stretched_exp(x, a, tau, p, c):
    """``a * exp(-(x/tau)**p) + c`` — the stretched (Kohlrausch) exponential.

    ``p`` names the noise that dominates the dephasing: ``p = 1`` is the
    Markovian/white-noise decay a plain exponential describes, ``p = 2`` the
    Gaussian decay of a 1/f-dominated environment. Fitting it FREE is only
    meaningful over a wide dynamic range — inside a fraction of one decay time
    every ``p`` fits equally well, which is why the log-spaced idle axis is what
    makes this parameter measurable at all.
    """
    x = np.asarray(x, dtype=float)
    # (x/tau)**p is real only for x >= 0; clip so an lmfit excursion into a
    # negative tau or a stray negative sample cannot return NaN and kill the fit.
    z = np.clip(x / tau, 0.0, None)
    return a * np.exp(-(z ** p)) + c


@register_fitter('stretched_exp')
class FitStretchedExponential(FunctionFitting):
    """
    Fit a stretched-exponential decay:
        a * exp(-(x / tau)**p) + c

    ``tau`` is the decay constant (a coherence time), ``p`` the stretch exponent
    (1 = exponential, 2 = Gaussian), ``c`` the asymptote.

    ``c`` IS NEEDED even when the physics says the decay reaches zero: this
    fitter's main consumer feeds it a lock-in MAGNITUDE ``|z|``, and a magnitude
    is Rician — at ``t >> tau`` it plateaus at the noise level instead of
    vanishing. Left unmodelled, that floor drags ``tau`` long.

    Set ``fix_p`` to freeze the exponent (``fix_p=1.0`` recovers a plain
    exponential with an offset). Accepts an ``xarray.DataArray`` with an ``'x'``
    coordinate, raw ``(x, y)`` arrays, or a bare ``y`` array, via the shared
    :func:`parse_xy` helper.
    """

    #: bounds on the stretch exponent. Below ~0.3 the model is numerically flat
    #: near the origin; above ~4 it is indistinguishable from a hard cutoff.
    P_BOUNDS = (0.3, 4.0)

    def __init__(self, data: DataArray = None, fix_p: float = None, x=None):
        self.fix_p = None if fix_p is None else float(fix_p)
        self._data_parser(data, x)
        self.model = Model(stretched_exp)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, a, tau, p, c):
        return stretched_exp(x, a, tau, p, c)

    @staticmethod
    def _tau_seed(x, y):
        """First 1/e crossing of ``y`` above its floor.

        A span-based seed (``x_span / 2``) is what :class:`FitExponentialDecay`
        uses and is fine on a linear axis, but this fitter's axis is usually
        LOG-spaced over decades, where half the span can be an order of
        magnitude off. The crossing is scale-free.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        span = float(abs(x[-1] - x[0])) or 1.0
        if x.size < 2:
            return span
        floor = float(np.min(y))
        env = y - floor
        env0 = float(env[0]) if env[0] > 0 else float(np.max(env))
        if env0 <= 0:
            return span / 2.0
        below = np.flatnonzero(env <= env0 / np.e)
        if below.size and below[0] > 0:
            return float(x[below[0]] - x[0]) or span / 2.0
        # never decayed to 1/e inside the window — tau is at least the span
        return span

    def guess(self):
        x, y = self.x, self.y

        a_guess = float(y[0] - y[-1])
        if a_guess > 0:
            a_dict = dict(value=a_guess, min=0.0, max=4 * a_guess)
        elif a_guess < 0:
            a_dict = dict(value=a_guess, min=4 * a_guess, max=0.0)
        else:
            a_dict = dict(value=float(np.ptp(y)) or 1.0)

        tau_seed = self._tau_seed(x, y)
        span = float(abs(x[-1] - x[0])) or 1.0
        # bounded well outside the window on both sides: a decay that is barely
        # resolved (tau >> span) must stay fittable, just with a large error.
        tau_dict = dict(value=tau_seed, min=span * 1e-4, max=span * 1e3)

        if self.fix_p is not None:
            p_dict = dict(value=self.fix_p, vary=False)
        else:
            p_dict = dict(value=1.0, min=self.P_BOUNDS[0], max=self.P_BOUNDS[1])

        c_guess = float(y[-1])
        c_halfwidth = abs(a_guess) or float(np.ptp(y)) or abs(c_guess) or 1.0
        c_dict = dict(value=c_guess, min=c_guess - c_halfwidth, max=c_guess + c_halfwidth)

        self.params = self.model.make_params(a=a_dict, tau=tau_dict, p=p_dict, c=c_dict)
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
