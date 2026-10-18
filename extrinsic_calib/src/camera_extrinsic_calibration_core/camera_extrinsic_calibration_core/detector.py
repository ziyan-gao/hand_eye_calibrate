"""Chessboard detection and board pose (PnP).

Board frame: origin at inner corner 0, x along columns, y along rows,
z = x cross y (right-handed). Points in metres.
Detection logic follows extrinsics_estimate.py (findChessboardCorners +
cornerSubPix + solvePnP) with origin disambiguation and PnP checks added.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .transforms import make_T, rotation_angle

SUPPORTED_DISTORTION = ('plumb_bob', 'rational_polynomial')


@dataclass(frozen=True)
class BoardConfig:
    columns: int
    rows: int
    square_size_mm: float

    def validate(self):
        if self.columns < 3 or self.rows < 3:
            return 'columns and rows must be >= 3 inner corners'
        if self.columns == self.rows:
            return 'square boards (columns == rows) have a 90 deg ambiguity; use columns != rows'
        if not (0.0 < self.square_size_mm < 1000.0):
            return 'square_size_mm must be in (0, 1000)'
        return ''

    @property
    def color_disambiguates(self):
        # 180 deg rotation swaps the corner-0 outer square colour iff columns+rows is odd
        return (self.columns + self.rows) % 2 == 1

    def object_points(self):
        objp = np.zeros((self.columns * self.rows, 3), np.float64)
        objp[:, :2] = np.mgrid[0:self.columns, 0:self.rows].T.reshape(-1, 2)
        return objp * (self.square_size_mm * 1e-3)


@dataclass
class Intrinsics:
    K: np.ndarray
    D: np.ndarray
    width: int
    height: int
    frame_id: str
    model: str

    @staticmethod
    def from_camera_info(msg):
        model = msg.distortion_model or 'plumb_bob'
        if model not in SUPPORTED_DISTORTION:
            return None, f'unsupported distortion model "{model}"'
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
            return None, 'invalid camera matrix'
        if msg.width == 0 or msg.height == 0:
            return None, 'camera_info has zero image size'
        return Intrinsics(K, D, int(msg.width), int(msg.height), msg.header.frame_id, model), ''

    def to_dict(self):
        return {'K': self.K.tolist(), 'D': self.D.tolist(), 'width': self.width,
                'height': self.height, 'frame_id': self.frame_id, 'model': self.model}

    @staticmethod
    def from_dict(d):
        return Intrinsics(np.array(d['K'], float), np.array(d['D'], float), int(d['width']),
                          int(d['height']), d['frame_id'], d['model'])


@dataclass
class Detection:
    ok: bool
    reason: str = ''
    corners: np.ndarray = None          # (N, 2) pixels, canonical order
    T_camera_from_board: np.ndarray = None
    reproj_rms_px: float = float('nan')
    per_point_err_px: np.ndarray = None
    warnings: list = field(default_factory=list)


def _patch_mean(gray, pt, half=2):
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = gray.shape
    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
        return None
    return float(gray[y - half:y + half + 1, x - half:x + half + 1].mean())


def canonicalize_origin(gray, corners, board):
    """Make corner 0 the corner whose outer (diagonal) square is dark.

    OpenCV may return the grid rotated by 180 deg between images. For boards with
    columns+rows odd the two orderings differ in the colour of the square outside
    corner 0, which fixes the order. Returns (corners, reason_if_failed).
    """
    c = corners.reshape(-1, 2)
    cols = board.columns

    def outer_vs_side(cc):
        c0, c1, cn = cc[0], cc[1], cc[cols]
        ex, ey = c1 - c0, cn - c0
        outer = _patch_mean(gray, c0 - 0.5 * ex - 0.5 * ey)
        side = _patch_mean(gray, c0 + 0.5 * ex - 0.5 * ey)
        if outer is None or side is None:
            return None
        return outer - side

    if not board.color_disambiguates:
        return c, ''  # caller adds a warning
    d = outer_vs_side(c)
    if d is None:
        return None, 'board border too close to image edge to resolve origin'
    if abs(d) < 15.0:
        return None, 'cannot resolve board origin (low square contrast)'
    if d > 0:  # outer square is light -> flip 180 deg
        c = c[::-1].copy()
    return c, ''


def detect_corners(gray, board, fast=False, fast_width=640):
    """fast=True: search on a downscaled image with CALIB_CB_FAST_CHECK (cheap rejection
    when no board is visible), then refine at full resolution with cornerSubPix."""
    pattern = (board.columns, board.rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    scale = 1.0
    search = gray
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
        if gray.shape[1] > fast_width:
            scale = fast_width / gray.shape[1]
            search = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, corners = cv2.findChessboardCorners(search, pattern, flags)
    if not ok or corners is None or len(corners) != board.columns * board.rows:
        return None, 'chessboard not detected'
    if scale != 1.0:
        corners = (corners / scale).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float64), ''


def _rms(objp, imgp, rvec, tvec, intr):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, intr.K, intr.D)
    err = np.linalg.norm(proj.reshape(-1, 2) - imgp, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), err


def solve_board_pose(corners, board, intr, ambiguity_ratio=1.2, ambiguity_deg=5.0):
    objp = board.object_points()
    imgp = corners.reshape(-1, 2)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(objp, imgp, intr.K, intr.D,
                                             flags=cv2.SOLVEPNP_IPPE)
    if n == 0:
        return Detection(False, 'solvePnP failed')
    cands = []
    for rv, tv in zip(rvecs, tvecs):
        rv, tv = cv2.solvePnPRefineLM(objp, imgp, intr.K, intr.D, rv.copy(), tv.copy())
        R, _ = cv2.Rodrigues(rv)
        pts_cam = (R @ objp.T).T + tv.reshape(1, 3)
        if np.any(pts_cam[:, 2] <= 0):
            continue
        rms, err = _rms(objp, imgp, rv, tv, intr)
        cands.append((rms, err, R, tv.reshape(3)))
    if not cands:
        return Detection(False, 'PnP: no solution with board in front of camera')
    cands.sort(key=lambda c: c[0])
    rms, err, R, t = cands[0]
    if len(cands) > 1:
        rms2, _, R2, _ = cands[1]
        if rms2 < ambiguity_ratio * max(rms, 1e-6) and \
                np.rad2deg(rotation_angle(R.T @ R2)) > ambiguity_deg:
            return Detection(False, 'PnP ambiguous (two planar solutions fit equally); tilt the board more')
    T = make_T(R, t)
    if not np.all(np.isfinite(T)):
        return Detection(False, 'PnP produced non-finite pose')
    return Detection(True, '', imgp, T, rms, err)


def detect_board(image_bgr, board, intr, fast=False):
    if image_bgr.shape[1] != intr.width or image_bgr.shape[0] != intr.height:
        return Detection(False, 'image size does not match camera_info')
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    corners, reason = detect_corners(gray, board, fast=fast)
    if corners is None:
        return Detection(False, reason)
    corners, reason = canonicalize_origin(gray, corners, board)
    if corners is None:
        return Detection(False, reason)
    det = solve_board_pose(corners, board, intr)
    if det.ok and not board.color_disambiguates:
        det.warnings.append('columns+rows even: origin may flip 180 deg between images')
    if det.ok:
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharp < 30.0:
            det.warnings.append(f'image may be blurred (laplacian var {sharp:.1f})')
    return det


def draw_overlay(image_bgr, det, board, intr):
    out = image_bgr.copy()
    if det is None or not det.ok:
        return out
    pts = det.corners.reshape(-1, 1, 2).astype(np.float32)
    cv2.drawChessboardCorners(out, (board.columns, board.rows), pts, True)
    T = det.T_camera_from_board
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    s = board.square_size_mm * 1e-3 * 3
    cv2.drawFrameAxes(out, intr.K, intr.D, rvec, T[:3, 3], s, 3)
    c0 = tuple(int(v) for v in det.corners[0])
    cv2.circle(out, c0, 8, (255, 255, 255), 2)
    cv2.putText(out, '0', (c0[0] + 10, c0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return out
