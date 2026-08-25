"""Complex lock-in over a swept PHASE (frame) axis — the phasor a frame sweep measures.

An experiment that sweeps the phase of its closing pulse through a full turn
measures, at every other sweep point, the qubit's equatorial Bloch vector as a
fringe in the frame. Projecting that fringe onto one cycle per turn collapses it
to a single complex number — the **phasor**::

    z = mean_frame( C * exp(-2j*pi*cycles*frame) )

with ``abs(z)`` the fringe CONTRAST (the coherence envelope) and ``angle(z)`` the
accumulated PHASE. For a known frequency (exactly one cycle per turn, by
construction of the sweep) this projection is the maximum-likelihood estimate of
both, needs no per-slice fit, and — fed the raw complex ``I + iQ`` — uses the full
IQ contrast regardless of the readout rotation, which enters only as a constant
phase offset.

TWO PROPERTIES ARE LOAD-BEARING, and both need an **endpoint-exclusive** uniform
frame grid (``np.linspace(0, 1, n, endpoint=False)``):

* **The readout offset cancels exactly.** A constant background ``C0`` contributes
  ``C0 * mean(exp(-2j*pi*frame)) = 0``. No baseline subtraction, no reference
  point, no stored blob centres.
* **Harmonics are rejected.** A contaminating ``m``-th harmonic of the fringe (an
  over-rotated pi/2 pulse produces ``m = 2``) survives only when
  ``m = cycles (mod n)``. A REAL contaminant carries both ``+m`` and ``-m``, so
  rejecting the 2f term needs ``n >= 4``: at ``n = 3`` the ``-2`` component
  aliases onto ``cycles = 1`` and leaks straight through. From ``n = 4`` the only
  survivors are ``m = 5, -3, 9, ...``.

Consumers: :mod:`scqat.estimators.ramsey_cryoscope` (takes ``angle``) and
:mod:`scqat.estimators.ramsey_phasor` (takes both). Pure math — no dataset,
estimator or parser imports.
"""

from typing import Union

import numpy as np


def lockin_phasor(
    values: Union[np.ndarray, "np.typing.ArrayLike"],
    frame: np.ndarray,
    *,
    axis: int = -1,
    cycles: int = 1,
) -> np.ndarray:
    """Project ``values`` onto ``cycles`` cycles per turn of the ``frame`` axis.

    Parameters
    ----------
    values : array_like
        The measured fringe, real or complex. One of its axes is the frame sweep.
    frame : 1-D array
        The frame axis in TURNS (0..1, endpoint-exclusive — see the module
        docstring for why the endpoint must be excluded). Its length must match
        ``values.shape[axis]``.
    axis : int, optional
        Which axis of ``values`` is the frame sweep (default: the last).
    cycles : int, optional
        Cycles per turn to demodulate at (default 1 — one closing-pulse phase
        turn produces exactly one fringe cycle).

    Returns
    -------
    complex ndarray
        ``values`` with ``axis`` contracted away: the complex phasor at every
        remaining point. ``abs`` is the fringe contrast, ``angle`` the phase.
    """
    values = np.asarray(values)
    frame = np.asarray(frame, dtype=float)
    if frame.ndim != 1:
        raise ValueError(f"frame must be 1-D, got shape {frame.shape}")
    if values.shape[axis] != frame.size:
        raise ValueError(
            f"frame has {frame.size} points but values.shape[{axis}] is "
            f"{values.shape[axis]} — the frame axis is not the one indexed."
        )
    if frame.size < 3:
        raise ValueError(
            f"a lock-in needs at least 3 frame points to separate the offset "
            f"from the two quadratures, got {frame.size}."
        )
    kernel = np.exp(-2j * np.pi * int(cycles) * frame)
    # broadcast the kernel along `axis` alone
    shape = [1] * values.ndim
    shape[axis] = frame.size
    return (values * kernel.reshape(shape)).mean(axis=axis)
