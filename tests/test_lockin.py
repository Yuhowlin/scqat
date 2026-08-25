"""Tests for the complex lock-in over a swept phase (frame) axis.

The two properties the estimators rely on -- exact cancellation of the readout
offset and rejection of the fringe's harmonics -- both hold only on an
endpoint-exclusive uniform frame grid, so they are pinned here rather than
rediscovered per consumer.
"""

import numpy as np
import pytest

from scqat.tools.lockin import lockin_phasor


def _frame(n):
    return np.linspace(0.0, 1.0, n, endpoint=False)


class TestLockinPhasor:
    def test_recovers_amplitude_and_phase(self):
        frame = _frame(16)
        for phi in (0.0, 0.7, -2.1, np.pi):
            fringe = 0.5 * np.cos(2 * np.pi * frame - phi)
            z = lockin_phasor(fringe, frame)
            # a real cosine of amplitude A projects to A/2
            assert abs(z) == pytest.approx(0.25)
            assert np.angle(z) == pytest.approx(-phi, abs=1e-9)

    def test_constant_offset_cancels_exactly(self):
        """A constant background contributes mean(exp(-2j*pi*frame)) = 0, so no
        baseline subtraction or reference point is ever needed."""
        frame = _frame(8)
        fringe = np.cos(2 * np.pi * frame)
        for offset in (0.0, 5.0, -1234.5):
            z = lockin_phasor(fringe + offset, frame)
            assert z == pytest.approx(lockin_phasor(fringe, frame))

    @pytest.mark.parametrize("n", [4, 5, 6, 16])
    def test_rejects_the_second_harmonic(self, n):
        """An over-rotated pi/2 pulse contaminates the fringe with m=2, which
        survives only when m == cycles (mod n)."""
        frame = _frame(n)
        clean = np.cos(2 * np.pi * frame)
        contaminated = clean + 0.4 * np.cos(2 * 2 * np.pi * frame)
        assert lockin_phasor(contaminated, frame) == pytest.approx(
            lockin_phasor(clean, frame)
        )

    def test_three_frames_alias_a_real_second_harmonic(self):
        """The rejection floor is FOUR frames, not three. A real 2f contaminant
        carries both +2 and -2, and at n=3 the -2 component lands on cycles=1
        (-2 mod 3 == 1) and leaks through at full weight. This is why the SCQO
        parameter floor is num_frames >= 4 rather than the lock-in's bare >= 3.
        """
        frame = _frame(3)
        clean = np.cos(2 * np.pi * frame)
        contaminated = clean + 0.4 * np.cos(2 * 2 * np.pi * frame)
        leak = abs(lockin_phasor(contaminated, frame) - lockin_phasor(clean, frame))
        assert leak == pytest.approx(0.2, rel=1e-6)
        # the pure +2 harmonic alone IS rejected at n=3 -- only the -2 aliases
        assert abs(lockin_phasor(np.exp(2j * np.pi * 2 * frame), frame)) == pytest.approx(
            0.0, abs=1e-12
        )
        assert abs(lockin_phasor(np.exp(-2j * np.pi * 2 * frame), frame)) == pytest.approx(
            1.0
        )

    def test_complex_input_is_blind_to_a_constant_rotation(self):
        frame = _frame(16)
        pop = 0.5 + 0.5 * np.cos(2 * np.pi * frame)
        base = lockin_phasor(complex(0.3, -0.2) + pop, frame)
        for theta in (0.0, 1.1, -2.3):
            rotated = lockin_phasor(complex(0.3, -0.2) + pop * np.exp(1j * theta), frame)
            assert abs(rotated) == pytest.approx(abs(base))
            assert np.angle(rotated) - np.angle(base) == pytest.approx(theta, abs=1e-9)

    def test_cycles_selects_the_harmonic(self):
        frame = _frame(16)
        fringe = np.cos(2 * 2 * np.pi * frame)
        assert abs(lockin_phasor(fringe, frame, cycles=1)) == pytest.approx(0.0, abs=1e-12)
        assert abs(lockin_phasor(fringe, frame, cycles=2)) == pytest.approx(0.5)

    def test_axis_contracts_the_named_axis_only(self):
        frame = _frame(8)
        values = np.random.default_rng(0).normal(size=(4, 8, 5))
        assert lockin_phasor(values, frame, axis=1).shape == (4, 5)
        moved = np.moveaxis(values, 1, -1)
        assert lockin_phasor(moved, frame, axis=-1).shape == (4, 5)
        np.testing.assert_allclose(
            lockin_phasor(values, frame, axis=1), lockin_phasor(moved, frame, axis=-1)
        )

    def test_rejects_a_non_1d_frame(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            lockin_phasor(np.ones((4, 4)), np.ones((2, 2)))

    def test_rejects_a_mismatched_axis(self):
        with pytest.raises(ValueError, match="not the one indexed"):
            lockin_phasor(np.ones((4, 8)), _frame(8), axis=0)

    def test_rejects_fewer_than_three_frame_points(self):
        """Below three points the offset and the two quadratures are not
        separable, so the projection would return a plausible wrong number."""
        with pytest.raises(ValueError, match="at least 3 frame points"):
            lockin_phasor(np.ones((4, 2)), _frame(2), axis=1)
