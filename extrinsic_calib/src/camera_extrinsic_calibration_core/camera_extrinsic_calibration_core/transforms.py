"""Rigid-transform helpers.

Naming: T_a_from_b maps homogeneous points expressed in frame b into frame a.
RPY follows the xArm SDK convention used in extrinsics_data_collection.py:
R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (fixed-axis XYZ).
"""
import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_rpy(R):
    """Inverse of rpy_to_matrix. Returns (roll, pitch, yaw, near_singular)."""
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)
    near_singular = abs(sp) > np.cos(np.deg2rad(1.0))
    if near_singular:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw), bool(near_singular)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def is_rotation(R, tol=1e-6):
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return False
    return (np.allclose(R.T @ R, np.eye(3), atol=tol)
            and abs(np.linalg.det(R) - 1.0) < tol)


def is_rigid(T, tol=1e-6):
    T = np.asarray(T, dtype=float)
    return (T.shape == (4, 4) and np.all(np.isfinite(T))
            and is_rotation(T[:3, :3], tol)
            and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


def orthonormalize(R):
    u, _, vt = np.linalg.svd(R)
    Rn = u @ vt
    if np.linalg.det(Rn) < 0:
        u[:, -1] *= -1
        Rn = u @ vt
    return Rn


def matrix_to_quat_xyzw(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def quat_xyzw_to_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_angle(R):
    """Geodesic angle (rad) of a rotation matrix."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def rotation_axis(R):
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(w)
    if n < 1e-9:
        # angle ~0 or ~pi; for pi use the dominant column of R + I
        M = R + np.eye(3)
        k = int(np.argmax(np.linalg.norm(M, axis=0)))
        v = M[:, k]
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-9 else np.array([0.0, 0.0, 1.0])
    return w / n


def mean_rotation(Rs):
    """Chordal L2 mean on SO(3) (projection of the arithmetic mean)."""
    return orthonormalize(np.sum(np.asarray(Rs), axis=0))


def xarm_pose_to_T(pose_mm_rad):
    """xArm RobotMsg.pose [x,y,z (mm), roll,pitch,yaw (rad)] -> T_base_from_tcp in m."""
    x, y, z, r, p, yw = [float(v) for v in pose_mm_rad[:6]]
    return make_T(rpy_to_matrix(r, p, yw), np.array([x, y, z]) * 1e-3)


def T_to_xyz_rpy_deg(T):
    """For display only: mm and degrees."""
    r, p, y, sing = matrix_to_rpy(T[:3, :3])
    return (T[:3, 3] * 1e3).tolist() + list(np.rad2deg([r, p, y])), sing
