import numpy as np
from PIL import Image

from pechabridge.mining.augment import JitterConfig, apply_test_time_augmentation


def test_augmentation_is_deterministic_per_patch_and_trial():
    base = Image.new("RGB", (96, 32), (240, 240, 240))
    cfg = JitterConfig(
        translate_px=3,
        scale_range=(0.98, 1.02),
        brightness=0.05,
        contrast=0.05,
        blur_sigma=0.3,
    )
    a = apply_test_time_augmentation(base, patch_id=123, trial_idx=2, base_seed=999, jitter=cfg)
    b = apply_test_time_augmentation(base, patch_id=123, trial_idx=2, base_seed=999, jitter=cfg)
    c = apply_test_time_augmentation(base, patch_id=123, trial_idx=3, base_seed=999, jitter=cfg)

    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    arr_c = np.asarray(c)
    assert np.array_equal(arr_a, arr_b)
    assert not np.array_equal(arr_a, arr_c)

