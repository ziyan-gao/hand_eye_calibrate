import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera_extrinsic_calibration_core import handeye  # noqa: E402
from camera_extrinsic_calibration_core.detector import (BoardConfig, Intrinsics,  # noqa: E402
                                                        detect_board)
from camera_extrinsic_calibration_core.session import Session  # noqa: E402
from camera_extrinsic_calibration_core.transforms import (  # noqa: E402
    inv_T, make_T, matrix_to_quat_xyzw, matrix_to_rpy, quat_xyzw_to_matrix, rotation_angle,
    rpy_to_matrix, xarm_pose_to_T)

rng = np.random.default_rng(0)


def rand_T(rot_deg=180, trans=0.5):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    R, _ = cv2.Rodrigues(axis * ang)
    return make_T(R, rng.uniform(-trans, trans, 3))


def assert_T_close(A, B, mm=1e-3, deg=1e-3):
    assert np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1e3 < mm
    assert np.rad2deg(rotation_angle(A[:3, :3].T @ B[:3, :3])) < deg


# ---------------- transforms
def test_rpy_roundtrip_and_sdk_convention():
    for _ in range(100):
        r, p, y = rng.uniform(-3, 3), rng.uniform(-1.5, 1.5), rng.uniform(-3, 3)
        R = rpy_to_matrix(r, p, y)
        # same composition as extrinsics_data_collection.euler_to_rotation_matrix
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        assert np.allclose(R, Rz @ Ry @ Rx)
        r2, p2, y2, _ = matrix_to_rpy(R)
        assert np.allclose(rpy_to_matrix(r2, p2, y2), R, atol=1e-9)


def test_quaternion_roundtrip():
    for _ in range(100):
        R = rand_T()[:3, :3]
        q = matrix_to_quat_xyzw(R)
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        assert np.allclose(quat_xyzw_to_matrix(q), R, atol=1e-9)


def test_xarm_pose_units():
    T = xarm_pose_to_T([100.0, -200.0, 300.0, 0.0, 0.0, np.pi / 2])
    assert np.allclose(T[:3, 3], [0.1, -0.2, 0.3])
    assert np.allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


# ---------------- hand-eye with known ground truth
def diverse_robot_poses(n):
    Bs = []
    for i in range(n):
        R = rpy_to_matrix(np.pi + rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
                          rng.uniform(-1.0, 1.0))
        Bs.append(make_T(R, [0.4 + rng.uniform(-0.1, 0.1), rng.uniform(-0.15, 0.15),
                             0.4 + rng.uniform(-0.1, 0.1)]))
    return Bs


@pytest.mark.parametrize('method', sorted(handeye.METHODS))
def test_eye_in_hand_recovers_X(method):
    X = make_T(rpy_to_matrix(0.05, -0.1, 1.6), [0.03, -0.06, 0.08])  # T_eef_from_camera
    Y = make_T(rpy_to_matrix(0.0, 0.0, 0.3), [0.5, 0.05, 0.0])      # T_base_from_board
    Bs = diverse_robot_poses(15)
    Cs = [inv_T(X) @ inv_T(B) @ Y for B in Bs]                      # T_camera_from_board
    res = handeye.solve('eye_in_hand', Bs, Cs, method)
    assert res.ok, res.message
    assert_T_close(res.X, X, mm=0.01, deg=0.01)
    assert_T_close(res.fixed_mean, Y, mm=0.01, deg=0.01)
    assert res.trans_max_mm < 0.01


@pytest.mark.parametrize('method', sorted(handeye.METHODS))
def test_eye_to_hand_recovers_X(method):
    X = make_T(rpy_to_matrix(2.5, 0.2, -0.4), [0.9, 0.2, 0.7])   # T_base_from_camera
    Z = make_T(rpy_to_matrix(0.0, 0.1, 0.0), [0.0, 0.02, 0.05])  # T_eef_from_board
    Bs = diverse_robot_poses(15)
    Cs = [inv_T(X) @ B @ Z for B in Bs]
    res = handeye.solve('eye_to_hand', Bs, Cs, method)
    assert res.ok, res.message
    assert_T_close(res.X, X, mm=0.01, deg=0.01)
    assert_T_close(res.fixed_mean, Z, mm=0.01, deg=0.01)


def test_wrong_direction_is_detected():
    """Solving eye-in-hand data as eye-to-hand must give large consistency error."""
    X = make_T(rpy_to_matrix(0.05, -0.1, 1.6), [0.03, -0.06, 0.08])
    Y = make_T(np.eye(3), [0.5, 0.0, 0.0])
    Bs = diverse_robot_poses(15)
    Cs = [inv_T(X) @ inv_T(B) @ Y for B in Bs]
    res = handeye.solve('eye_to_hand', Bs, Cs, 'TSAI')
    assert (not res.ok) or res.trans_rms_mm > 5.0


def test_noisy_data_reasonable():
    X = make_T(rpy_to_matrix(0.05, -0.1, 1.6), [0.03, -0.06, 0.08])
    Y = make_T(np.eye(3), [0.5, 0.0, 0.0])
    Bs = diverse_robot_poses(20)
    Cs = []
    for B in Bs:
        C = inv_T(X) @ inv_T(B) @ Y
        Cs.append(C @ make_T(cv2.Rodrigues(rng.normal(0, 1e-3, 3))[0], rng.normal(0, 3e-4, 3)))
    res = handeye.solve('eye_in_hand', Bs, Cs, 'PARK')
    assert res.ok
    assert_T_close(res.X, X, mm=3.0, deg=0.5)


# ---------------- degenerate motion
def test_pure_translation_rejected():
    R = rpy_to_matrix(np.pi, 0, 0)
    Bs = [make_T(R, [0.4 + 0.05 * i, 0.02 * i, 0.3]) for i in range(10)]
    mc = handeye.check_motion(Bs)
    assert not mc.ok and 'degenerate' in mc.errors[0]


def test_coaxial_rotation_rejected():
    Bs = [make_T(rpy_to_matrix(np.pi, 0, 0.3 * i), [0.4, 0.0, 0.3 + 0.01 * i])
          for i in range(10)]
    mc = handeye.check_motion(Bs)
    assert not mc.ok and 'parallel' in mc.errors[0]


def test_too_few_and_duplicates():
    B = diverse_robot_poses(1)[0]
    mc = handeye.check_motion([B, B])
    assert not mc.ok
    mc = handeye.check_motion([B, B] + diverse_robot_poses(15))
    assert any('near-identical' in w for w in mc.warnings)


def test_diverse_motion_ok():
    assert handeye.check_motion(diverse_robot_poses(15)).ok


# ---------------- detector on a rendered board
def render_board(board, K, T_cam_board, size=(1280, 720)):
    s = board.square_size_mm * 1e-3
    W, H = board.columns + 1, board.rows + 1
    ppm = 2000  # pixels per metre on the texture
    margin = int(0.03 * ppm)
    tex = np.full((int(H * s * ppm) + 2 * margin, int(W * s * ppm) + 2 * margin), 255, np.uint8)
    for j in range(H):
        for i in range(W):
            if (i + j) % 2 == 0:  # square diagonally outside corner 0 is dark
                x0, y0 = margin + int(i * s * ppm), margin + int(j * s * ppm)
                tex[y0:y0 + int(s * ppm), x0:x0 + int(s * ppm)] = 0
    # texture pixel (u,v) -> board coords: x = (u - margin)/ppm - s, y likewise
    A = np.array([[1 / ppm, 0, -margin / ppm - s], [0, 1 / ppm, -margin / ppm - s], [0, 0, 1]])
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
    return cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_LINEAR, borderValue=128)


@pytest.mark.parametrize('yaw', [0.0, np.pi, 0.3, np.pi + 0.3])
def test_detector_origin_is_consistent(yaw):
    board = BoardConfig(9, 6, 26.5)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    intr = Intrinsics(K, np.zeros(5), 1280, 720, 'camera_color_optical_frame', 'plumb_bob')
    center = np.array([4 * 0.0265, 2.5 * 0.0265, 0])
    R = rpy_to_matrix(0.25, -0.2, yaw)
    t = np.array([0.0, 0.0, 0.45]) - R @ center
    T_true = make_T(R, t)
    img = cv2.cvtColor(render_board(board, K, T_true), cv2.COLOR_GRAY2BGR)
    det = detect_board(img, board, intr)
    assert det.ok, det.reason
    # rotated 180 deg in the image, the origin must still be the same physical corner
    assert_T_close(det.T_camera_from_board, T_true, mm=1.0, deg=0.3)
    assert det.reproj_rms_px < 0.5


def test_detector_rejects_size_mismatch():
    board = BoardConfig(9, 6, 26.5)
    intr = Intrinsics(np.eye(3), np.zeros(5), 640, 480, 'f', 'plumb_bob')
    assert not detect_board(np.zeros((720, 1280, 3), np.uint8), board, intr).ok


def test_board_validation():
    assert BoardConfig(9, 6, 26.5).validate() == ''
    assert BoardConfig(7, 7, 20).validate() != ''
    assert BoardConfig(9, 6, 0).validate() != ''


# ---------------- session lifecycle
def test_session_add_delete_reload(tmp_path):
    s = Session.create(str(tmp_path), 'test', {'columns': 9, 'rows': 6, 'square_size_mm': 26.5,
                                               'mode': 'eye_in_hand'}, {'eef': 'tcp'}, {})
    img = np.zeros((10, 10, 3), np.uint8)
    a = s.add_sample({'reproj_rms_px': 0.1, 'dt_ms': 1.0}, img)
    b = s.add_sample({'reproj_rms_px': 0.2, 'dt_ms': 2.0}, img)
    assert s.dataset_revision == 2
    t, _ = s.exclude('last')
    assert t['sample_id'] == b['sample_id'] and s.dataset_revision == 3
    t, _ = s.exclude('last')
    assert t['sample_id'] == a['sample_id']
    assert s.exclude('last')[0] is None
    c = s.add_sample({'reproj_rms_px': 0.3, 'dt_ms': 3.0}, img)
    assert c['sequence'] == 3 and c['sample_id'] not in (a['sample_id'], b['sample_id'])
    s2 = Session.load(s.dir)
    assert [x['sample_id'] for x in s2.valid_samples()] == [c['sample_id']]
    assert s2.dataset_revision == s.dataset_revision
    p = s.save_result('r1', False, {'x': 1}, {'y': 2})
    with pytest.raises(FileExistsError):
        s.save_result('r1', False, {'x': 1}, {'y': 2})
    assert os.path.isfile(os.path.join(p, 'extrinsics.yaml'))
