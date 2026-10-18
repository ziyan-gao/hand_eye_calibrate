"""OpenCV hand-eye solving, motion checks and consistency report.

B_i = T_base_from_eef, C_i = T_camera_from_board.
eye_in_hand : X = T_eef_from_camera,  Y_i = B_i @ X @ C_i      (fixed board in base)
eye_to_hand : X = T_base_from_camera, Z_i = inv(B_i) @ X @ C_i (board rigid on eef)
"""
import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from .transforms import inv_T, is_rigid, make_T, mean_rotation, rotation_angle, rotation_axis

METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}
MODES = ('eye_in_hand', 'eye_to_hand')


@dataclass
class MotionCheck:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    max_rel_rot_deg: float = 0.0
    max_axis_spread_deg: float = 0.0


def check_motion(B_list, min_samples=3, min_rel_rot_deg=10.0, min_axis_spread_deg=20.0,
                 dup_rot_deg=1.0, dup_trans_mm=5.0, recommended_samples=15):
    res = MotionCheck(True)
    n = len(B_list)
    if n < min_samples:
        res.errors.append(f'need at least {min_samples} valid samples, have {n}')
    axes = []
    for i, j in itertools.combinations(range(n), 2):
        Rij = B_list[i][:3, :3].T @ B_list[j][:3, :3]
        ang = np.rad2deg(rotation_angle(Rij))
        dt = np.linalg.norm(B_list[i][:3, 3] - B_list[j][:3, 3]) * 1e3
        res.max_rel_rot_deg = max(res.max_rel_rot_deg, ang)
        if ang < dup_rot_deg and dt < dup_trans_mm:
            res.warnings.append(f'samples {i} and {j} have near-identical robot poses')
        if ang >= min_rel_rot_deg:
            axes.append(rotation_axis(Rij))
    for a, b in itertools.combinations(axes, 2):
        spread = np.rad2deg(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0)))
        res.max_axis_spread_deg = max(res.max_axis_spread_deg, spread)
    if n >= min_samples:
        if not axes:
            res.errors.append(f'no relative rotation >= {min_rel_rot_deg:.0f} deg '
                              '(pure translation or tiny rotations are degenerate)')
        elif res.max_axis_spread_deg < min_axis_spread_deg:
            res.errors.append('relative rotation axes are (near) parallel: '
                              f'max spread {res.max_axis_spread_deg:.1f} deg < '
                              f'{min_axis_spread_deg:.0f} deg; rotate about another axis')
    if not res.errors and n < recommended_samples:
        res.warnings.append(f'only {n} samples; {recommended_samples}-25 diverse poses recommended')
    res.ok = not res.errors
    return res


@dataclass
class HandEyeResult:
    ok: bool
    message: str = ''
    X: np.ndarray = None
    fixed_mean: np.ndarray = None       # mean Y (eye_in_hand) or Z (eye_to_hand)
    per_sample_trans_mm: list = field(default_factory=list)
    per_sample_rot_deg: list = field(default_factory=list)
    trans_rms_mm: float = float('nan')
    trans_max_mm: float = float('nan')
    rot_rms_deg: float = float('nan')
    rot_max_deg: float = float('nan')


def chain(mode, B, X, C):
    if mode == 'eye_in_hand':
        return B @ X @ C
    return inv_T(B) @ X @ C


def consistency(mode, B_list, X, C_list):
    fixed = [chain(mode, B, X, C) for B, C in zip(B_list, C_list)]
    Rm = mean_rotation([F[:3, :3] for F in fixed])
    tm = np.mean([F[:3, 3] for F in fixed], axis=0)
    dt = [float(np.linalg.norm(F[:3, 3] - tm) * 1e3) for F in fixed]
    dr = [float(np.rad2deg(rotation_angle(Rm.T @ F[:3, :3]))) for F in fixed]
    return make_T(Rm, tm), dt, dr


def solve(mode, B_list, C_list, method='TSAI'):
    if mode not in MODES:
        return HandEyeResult(False, f'unknown mode {mode}')
    if method not in METHODS:
        return HandEyeResult(False, f'unknown method {method}; choose {sorted(METHODS)}')
    if len(B_list) != len(C_list) or len(B_list) < 3:
        return HandEyeResult(False, 'need >= 3 paired samples')
    A = B_list if mode == 'eye_in_hand' else [inv_T(B) for B in B_list]
    try:
        R, t = cv2.calibrateHandEye(
            [a[:3, :3] for a in A], [a[:3, 3].reshape(3, 1) for a in A],
            [c[:3, :3] for c in C_list], [c[:3, 3].reshape(3, 1) for c in C_list],
            method=METHODS[method])
    except cv2.error as e:
        return HandEyeResult(False, f'calibrateHandEye failed: {e}')
    X = make_T(np.asarray(R, float), np.asarray(t, float).reshape(3))
    if not is_rigid(X, tol=1e-4):
        return HandEyeResult(False, 'solver returned non-finite or invalid rotation')
    fixed, dt, dr = consistency(mode, B_list, X, C_list)
    res = HandEyeResult(True, '', X, fixed, dt, dr)
    res.trans_rms_mm = float(np.sqrt(np.mean(np.square(dt))))
    res.trans_max_mm = float(np.max(dt))
    res.rot_rms_deg = float(np.sqrt(np.mean(np.square(dr))))
    res.rot_max_deg = float(np.max(dr))
    return res


def compare_methods(mode, B_list, C_list, reference_X):
    """Solve with every OpenCV method and report deviation from reference_X.

    Agreement between methods is not proof of accuracy; disagreement flags
    ill-conditioned data or a method failing on it (e.g. TSAI near 180 deg rotations).
    """
    out = {}
    for name in METHODS:
        r = solve(mode, B_list, C_list, name)
        if not r.ok:
            out[name] = {'ok': False, 'message': r.message}
            continue
        E = inv_T(reference_X) @ r.X
        out[name] = {'ok': True,
                     'diff_trans_mm': float(np.linalg.norm(E[:3, 3]) * 1e3),
                     'diff_rot_deg': float(np.rad2deg(rotation_angle(E[:3, :3]))),
                     'consistency_trans_rms_mm': r.trans_rms_mm}
    return out
