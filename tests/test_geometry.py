import numpy as np
import pytest

from kavach.geometry import foot_point, head_region, ioa, iou, point_in_polygon, torso_angle_deg


def kpts_with(points: dict[int, tuple[float, float]]) -> np.ndarray:
    k = np.zeros((17, 3), dtype=np.float32)
    for i, (x, y) in points.items():
        k[i] = (x, y, 0.9)
    return k


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_ioa_small_box_inside_large_box_is_one():
    helmet, head = (10, 10, 20, 20), (0, 0, 100, 100)
    assert ioa(helmet, head) == pytest.approx(1.0)
    assert iou(helmet, head) < 0.05  # why IoA, not IoU, is used for PPE association


def test_foot_point_uses_ankles_then_falls_back_to_box_bottom():
    box = (0, 0, 100, 200)
    assert foot_point(box, kpts_with({15: (40, 190), 16: (60, 194)})) == pytest.approx((50, 192))
    assert foot_point(box, None) == pytest.approx((50, 200))


def test_torso_angle_upright_and_lying():
    upright = kpts_with({5: (45, 50), 6: (55, 50), 11: (45, 120), 12: (55, 120)})
    lying = kpts_with({5: (50, 45), 6: (50, 55), 11: (120, 45), 12: (120, 55)})
    assert torso_angle_deg(upright) == pytest.approx(0, abs=1)
    assert torso_angle_deg(lying) == pytest.approx(90, abs=1)
    assert torso_angle_deg(kpts_with({5: (0, 0)})) is None


def test_head_region_from_face_keypoints_sits_above_the_eyes():
    k = kpts_with({0: (50, 40), 1: (45, 35), 2: (55, 35), 5: (40, 70), 6: (60, 70)})
    r = head_region((20, 20, 80, 200), k)
    assert r[1] < 35 and r[0] < 45 and r[2] > 55


def test_point_in_polygon_includes_boundary():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
    assert point_in_polygon((5, 5), poly)
    assert point_in_polygon((10, 5), poly)
    assert not point_in_polygon((15, 5), poly)
