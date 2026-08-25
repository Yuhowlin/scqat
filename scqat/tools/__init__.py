"""
tools package — shared mathematical fitters and helpers.

Importing this package eagerly imports each fitter module so that they
register with the ``get_fitter`` factory. New fitters must be added here
to remain discoverable.
"""

from .function_fitting import FunctionFitting, register_fitter, get_fitter
from .dip_fit import DIP_KNOBS, DIP_METHODS, fit_dip, validate_dip_kwargs
from .dip_finder import (
    DIP_FINDER_KNOBS,
    find_resonator_dips,
    validate_dip_finder_kwargs,
)
from .peak_fit import PEAK_KNOBS, fit_peaks, validate_peak_kwargs
from .peak_map import track_peaks
from .iq_reduce import (
    AXIAL_KNOBS,
    IQ_REDUCE_KNOBS,
    RADIAL_KNOBS,
    axial,
    axis_angle,
    ground_ref,
    radial,
    validate_iq_reduce_kwargs,
)
from .discriminate import (
    DISCRIMINATE_KNOBS,
    discriminate_states,
    validate_discriminate_kwargs,
)
from .ge_discriminator import compute_ge_discriminator
from .ramsey_fit import RAMSEY_MODELS, fit_ramsey
from .lockin import lockin_phasor
from .flux_predistortion import (
    MIN_A_DC,
    exp_sum_step_response,
    exp_sum_to_cascade,
    partition_exp_stages,
)
from .step_response_fit import (
    fit_step_response,
    sequential_exp_fit,
    single_exp_decay,
)
from .telegraph_psd import (
    MIN_PSD_CONTRAST,
    TELEGRAPH_PSD_KNOBS,
    fit_telegraph_psd,
    lorentzian_knee,
    telegraph_spectrum,
    validate_telegraph_psd_kwargs,
)
from .timeseries_psd import (
    TIMESERIES_PSD_KNOBS,
    timeseries_psd,
    validate_timeseries_psd_kwargs,
)
from .allan import overlapping_allan_deviation
from .ade_decay import (
    ADE_BOOTSTRAP_MIN_FINITE,
    ade_bootstrap_sigma_t1,
    ade_gamma,
    ade_sigma_gamma,
)
from . import fit_abscos  # noqa: F401
from . import fit_cosine  # noqa: F401
from . import fit_damped_oscillation  # noqa: F401
from . import fit_damping_beat  # noqa: F401
from . import fit_exp_decay  # noqa: F401
from . import fit_multi_damped_oscillation  # noqa: F401
from . import fit_gaussian2d  # noqa: F401
from . import fit_lorentzian  # noqa: F401
from . import fit_lorentzian_bg  # noqa: F401
from . import fit_notch_circle  # noqa: F401
from . import fit_powerlaw_base  # noqa: F401
from . import fit_qubit_decoherence  # noqa: F401
from . import fit_stretched_exp  # noqa: F401
from . import fit_transmon_freq_flux  # noqa: F401
