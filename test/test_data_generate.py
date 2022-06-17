"""Test data generation."""
import numpy as np


def generate_saw(length: int,
                 period: int,
                 amplitude: float) -> np.ndarray:
    """Generate saw signal with specified length, period and amplitude."""
    saw = np.empty(shape=(length, ))
    period -= 1
    amplitude = amplitude / period

    for num, left in enumerate(range(0, length, period)):
        if num % 2 == 0:
            saw[left:left + period] = np.arange(0, period) * amplitude
        else:
            saw[left:left + period] = np.arange(period, 0, -1) * amplitude
    return saw


def test_generate_saw():
    """Test saw generation."""
    saw = generate_saw(12, 3, 6)
    expected = np.array([0, 3, 6, 3, 0, 3, 6, 3, 0, 3, 6, 3])
    assert np.all(np.equal(saw, expected)), "Saw not generated as expected."
