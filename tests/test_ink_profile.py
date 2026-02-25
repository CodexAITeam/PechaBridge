import numpy as np

from pechabridge.data_gen.ink_profile import detect_profile_minima, smooth_profile


def test_detect_profile_minima_on_synthetic_profile():
    x = np.arange(0, 120, dtype=np.float32)
    profile = np.full_like(x, 12.0, dtype=np.float32)
    # Three valleys.
    profile -= 8.0 * np.exp(-((x - 20.0) ** 2) / (2.0 * 2.5**2))
    profile -= 7.5 * np.exp(-((x - 55.0) ** 2) / (2.0 * 3.0**2))
    profile -= 8.5 * np.exp(-((x - 90.0) ** 2) / (2.0 * 2.8**2))

    smoothed = smooth_profile(profile, sigma=1.5)
    minima = detect_profile_minima(smoothed, min_dist_px=12, prominence=1.5)

    assert minima.size >= 3
    assert np.min(np.abs(minima - 20)) <= 3
    assert np.min(np.abs(minima - 55)) <= 4
    assert np.min(np.abs(minima - 90)) <= 4

