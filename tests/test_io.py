import numpy as np

from ffr.preprocess import signed_distance


def test_signed_distance_sign_and_scale():
    m = np.zeros((40, 40, 40), bool)
    m[10:30, 10:30, 10:30] = True
    s = signed_distance(m, 2.0)
    assert s[20, 20, 20] < 0 and s[0, 0, 0] > 0
    assert abs(s[20, 20, 2] - 16.0) < 1e-6  # 8 voxels outside at 2 mm
