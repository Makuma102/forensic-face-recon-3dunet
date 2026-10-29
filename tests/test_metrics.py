import numpy as np

from ffr.metrics import sdf_to_mesh, surface_metrics, identification
from ffr.preprocess import signed_distance


def _sphere(r, shape=(64, 64, 64), sp=1.0, c=(32, 32, 32)):
    z, y, x = np.indices(shape)
    return (np.sqrt((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) * sp) <= r


ROI = dict(y_max_mm=64, z_min_mm=0, z_max_mm=64, border_mm=0.5, size_mm=(64, 64, 64))


def test_identical_surfaces_have_near_zero_error():
    m = sdf_to_mesh(signed_distance(_sphere(20), 1.0), 1.0)
    r = surface_metrics(m, m, ROI, n_samples=50_000)
    assert r["mean_mm"] < 0.3


def test_concentric_spheres_error_equals_radius_gap():
    a = sdf_to_mesh(signed_distance(_sphere(20), 1.0), 1.0)
    b = sdf_to_mesh(signed_distance(_sphere(23), 1.0), 1.0)
    r = surface_metrics(b, a, ROI, gt_sdf_fine=signed_distance(_sphere(20), 1.0), n_samples=50_000)
    assert abs(r["mean_mm"] - 3.0) < 0.5
    assert r["signed_mean_mm"] > 2.0  # predicted surface lies outside the target


def test_identification_picks_matching_shape():
    shapes = {k: sdf_to_mesh(signed_distance(_sphere(r), 1.0), 1.0) for k, r in zip("abc", (14, 19, 24))}
    res = identification(shapes, shapes, ROI, n_points=800, icp=False)
    assert res["rank1"] == 1.0
