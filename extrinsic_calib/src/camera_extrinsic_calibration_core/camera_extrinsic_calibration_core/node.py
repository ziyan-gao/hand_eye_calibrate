"""Calibration backend node.

Inputs : color image + camera_info (realsense-ros), xarm RobotMsg (/xarm/robot_states).
Outputs: /calibration/status, /calibration/preview, /calibration/eef_transform,
         /calibration/board_transform, services and the Solve action.
"""
import collections
import os
import platform
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Transform, TransformStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

# Large images (~2.7 MB at 1280x720) are split into many UDP fragments; with BEST_EFFORT one
# lost fragment drops the frame. RELIABLE + depth 1 keeps only the newest frame.
IMAGE_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE)
from rclpy.time import Time
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener
import yaml
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool
from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import SetInt16, SetInt16ById

from camera_extrinsic_calibration_interfaces.action import Solve
from camera_extrinsic_calibration_interfaces.msg import (CalibrationResult, CalibrationStatus,
                                                         SampleSummary)
from camera_extrinsic_calibration_interfaces.srv import (Capture, Configure, DeleteSample,
                                                         ListSamples, LoadSession, SaveResult)

from . import handeye
from .detector import BoardConfig, Intrinsics, detect_board, draw_overlay
from .session import Session, T_to_list, list_to_T
from .transforms import (make_T, matrix_to_quat_xyzw, quat_xyzw_to_matrix, rotation_angle,
                         xarm_pose_to_T)

S = CalibrationStatus


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_stamp(t):
    return Time(nanoseconds=int(round(t * 1e9))).to_msg()


def T_to_msg(T):
    m = Transform()
    m.translation.x, m.translation.y, m.translation.z = [float(v) for v in T[:3, 3]]
    q = matrix_to_quat_xyzw(T[:3, :3])
    m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w = [float(v) for v in q]
    return m


class RobotSample:
    __slots__ = ('stamp', 'recv', 'T', 'pose', 'offset', 'state', 'mode', 'err')

    def __init__(self, msg, recv):
        self.stamp = stamp_to_sec(msg.header.stamp)
        self.recv = recv
        self.pose = [float(v) for v in msg.pose]
        self.offset = [float(v) for v in msg.offset]
        self.T = xarm_pose_to_T(self.pose)
        self.state = int(msg.state)
        self.mode = int(msg.mode)
        self.err = int(msg.err)


class Frame:
    __slots__ = ('stamp', 'recv', 'image', 'frame_id', 'seq')

    def __init__(self, stamp, recv, image, frame_id, seq):
        self.stamp, self.recv, self.image, self.frame_id, self.seq = stamp, recv, image, frame_id, seq


class CalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_extrinsic_calibration')
        p = self.declare_parameter
        self.image_topic = p('image_topic', '/camera/camera/color/image_raw').value
        self.info_topic = p('camera_info_topic', '/camera/camera/color/camera_info').value
        self.robot_topic = p('robot_states_topic', '/xarm/robot_states').value
        self.base_frame = p('base_frame', 'link_base').value
        self.eef_frame = p('eef_frame', 'link_tcp').value
        self.board_frame = p('board_frame', 'calibration_board').value
        self.data_root = p('data_root', '/data/calibration').value
        # TCP offset (mm, rad) the controller must report for the parallel-gripper TCP
        self.check_tcp = p('check_tcp_offset', True).value
        self.expected_tcp = list(p('expected_tcp_offset', [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]).value)
        self.tcp_tol_mm = p('tcp_offset_tol_mm', 0.5).value
        self.tcp_tol_rad = p('tcp_offset_tol_rad', 0.002).value
        # Engineering defaults from the spec; not validated acceptance thresholds
        self.capture_timeout = p('capture_timeout_s', 2.0).value
        self.static_window = p('static_window_s', 1.0).value
        self.max_dt = p('max_pair_dt_ms', 120.0).value * 1e-3
        self.freshness = p('freshness_s', 1.0).value
        self.static_trans_mm = p('static_trans_mm', 0.5).value
        self.static_rot_deg = p('static_rot_deg', 0.1).value
        self.max_reproj_px = p('max_reproj_px', 2.0).value
        self.min_rel_rot_deg = p('min_rel_rot_deg', 10.0).value
        self.min_axis_spread_deg = p('min_axis_spread_deg', 20.0).value
        self.method_warn_mm = p('method_disagreement_warn_mm', 2.0).value
        self.method_warn_deg = p('method_disagreement_warn_deg', 0.5).value
        # Publish the solved camera pose to /tf_static as parent -> camera_link
        self.publish_tf = p('publish_tf', True).value
        self.camera_link_frame = p('camera_link_frame', 'camera_link').value
        # Manual (teaching) mode lets the operator drag the arm between captures
        self.xarm_ns = p('xarm_ns', '/xarm').value.rstrip('/')
        self.manual_on_start = p('manual_mode_on_start', True).value
        self.manual_start_done = False
        self.preview_width = p('preview_width', 640).value
        preview_hz = p('preview_rate_hz', 10.0).value
        status_hz = p('status_rate_hz', 10.0).value

        os.makedirs(self.data_root, exist_ok=True)
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.robot_buf = collections.deque(maxlen=2000)
        self.intr = None
        self.intr_reason = 'no camera_info received'
        self.latest_frame = None
        self.frame_seq = 0
        self.processed_seq = -1
        self.live_det = None
        self.live_det_stamp = 0.0
        self.live_det_recv = 0.0
        self.preview_seq = -1
        self.capture_click = None
        self.capture_frame = None
        self.capture_event = threading.Event()
        self.board = None
        self.mode = None
        self.session = None
        self.state = S.UNCONFIGURED
        self.busy = False
        self.message = 'configure board and mode'
        self.result = None          # dict with HandEyeResult and provenance
        self.result_failed = False

        g_img = MutuallyExclusiveCallbackGroup()
        g_in = MutuallyExclusiveCallbackGroup()
        g_timer = MutuallyExclusiveCallbackGroup()
        g_detect = MutuallyExclusiveCallbackGroup()
        g_status = MutuallyExclusiveCallbackGroup()
        g_srv = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_published = None    # (parent, child) we currently own

        self.create_subscription(Image, self.image_topic, self.on_image,
                                 IMAGE_QOS, callback_group=g_img)
        self.create_subscription(CameraInfo, self.info_topic, self.on_info,
                                 qos_profile_sensor_data, callback_group=g_in)
        self.create_subscription(RobotMsg, self.robot_topic, self.on_robot, 50,
                                 callback_group=g_in)
        self.pub_status = self.create_publisher(CalibrationStatus, '/calibration/status', 10)
        self.pub_preview = self.create_publisher(Image, '/calibration/preview', IMAGE_QOS)
        self.pub_eef = self.create_publisher(TransformStamped, '/calibration/eef_transform', 10)
        self.pub_board = self.create_publisher(TransformStamped, '/calibration/board_transform', 10)

        self.create_service(Configure, '/calibration/configure', self.srv_configure,
                            callback_group=g_srv)
        self.create_service(Capture, '/calibration/capture', self.srv_capture,
                            callback_group=g_srv)
        self.create_service(ListSamples, '/calibration/list_samples', self.srv_list,
                            callback_group=g_srv)
        self.create_service(DeleteSample, '/calibration/delete_sample', self.srv_delete,
                            callback_group=g_srv)
        self.create_service(SaveResult, '/calibration/save_result', self.srv_save,
                            callback_group=g_srv)
        self.create_service(LoadSession, '/calibration/load_session', self.srv_load,
                            callback_group=g_srv)
        self.create_service(SetBool, '/calibration/set_manual_mode', self.srv_manual,
                            callback_group=g_srv)
        g_xarm = ReentrantCallbackGroup()
        self.cli_mode = self.create_client(SetInt16, f'{self.xarm_ns}/set_mode',
                                           callback_group=g_xarm)
        self.cli_state = self.create_client(SetInt16, f'{self.xarm_ns}/set_state',
                                            callback_group=g_xarm)
        self.cli_enable = self.create_client(SetInt16ById, f'{self.xarm_ns}/motion_enable',
                                             callback_group=g_xarm)
        self.create_timer(1.0, self.manual_on_start_tick, callback_group=g_srv)
        self.solve_server = ActionServer(
            self, Solve, '/calibration/solve', execute_callback=self.exec_solve,
            goal_callback=self.goal_solve, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=g_srv)

        self.create_timer(1.0 / preview_hz, self.publish_preview, callback_group=g_timer)
        self.create_timer(1.0 / preview_hz, self.detect_tick, callback_group=g_detect)
        self.create_timer(1.0 / status_hz, self.publish_status, callback_group=g_status)
        self.get_logger().info(f'image={self.image_topic} robot={self.robot_topic} '
                               f'data_root={self.data_root}')

    # ------------------------------------------------------------------ inputs
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_info(self, msg):
        intr, reason = Intrinsics.from_camera_info(msg)
        with self.lock:
            self.intr, self.intr_reason = intr, reason

    def on_robot(self, msg):
        if len(msg.pose) < 6:
            return
        rs = RobotSample(msg, self.now_s())
        with self.lock:
            self.robot_buf.append(rs)

    def on_image(self, msg):
        recv = self.now_s()
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'image conversion failed: {e}', throttle_duration_sec=5.0)
            return
        with self.lock:
            self.frame_seq += 1
            fr = Frame(stamp_to_sec(msg.header.stamp), recv, img, msg.header.frame_id,
                       self.frame_seq)
            self.latest_frame = fr
            if self.capture_click is not None and self.capture_frame is None \
                    and recv > self.capture_click:
                self.capture_frame = fr
                self.capture_event.set()

    # ------------------------------------------------------------------ helpers
    def tcp_offset_error(self, offset):
        if not self.check_tcp:
            return ''
        if len(offset) < 6:
            return 'robot_states has no TCP offset'
        d = np.abs(np.array(offset[:6]) - np.array(self.expected_tcp[:6]))
        if np.any(d[:3] > self.tcp_tol_mm) or np.any(d[3:] > self.tcp_tol_rad):
            return (f'controller TCP offset {np.round(offset[:6], 3).tolist()} != expected '
                    f'gripper TCP {self.expected_tcp}')
        return ''

    def robot_live(self, now):
        with self.lock:
            rs = self.robot_buf[-1] if self.robot_buf else None
        if rs is None:
            return None, 'no robot_states received'
        if now - rs.recv > self.freshness:
            return None, f'robot_states stale ({now - rs.recv:.2f} s)'
        if rs.err != 0:
            return None, f'robot error code {rs.err}'
        return rs, ''

    def board_cfg_dict(self):
        return {'columns': self.board.columns, 'rows': self.board.rows,
                'square_size_mm': self.board.square_size_mm, 'mode': self.mode,
                'board_origin': 'inner corner 0; outer square at corner 0 is dark',
                'eef_definition': 'xArm TCP (controller TCP offset = parallel gripper TCP)',
                'expected_tcp_offset_mm_rad': self.expected_tcp}

    def frames_dict(self):
        cam = self.intr.frame_id if self.intr else ''
        return {'base': self.base_frame, 'eef': self.eef_frame, 'camera_optical': cam}

    def versions(self):
        return {'opencv': cv2.__version__, 'numpy': np.__version__,
                'python': platform.python_version()}

    def result_state(self):
        if self.result is None:
            return S.RESULT_FAILED if self.result_failed else S.RESULT_NONE
        if self.session is None or self.result['session_id'] != self.session.id \
                or self.result['config_revision'] != self.session.config_revision \
                or self.result['dataset_revision'] != self.session.dataset_revision:
            return S.RESULT_STALE
        return S.RESULT_VALID

    def set_busy(self, busy, state=None, message=None):
        with self.lock:
            self.busy = busy
            if state is not None:
                self.state = state
            if message is not None:
                self.message = message

    # ------------------------------------------------------------------ live preview
    def detect_tick(self):
        """Board detection on the newest frame; runs in its own thread so a slow
        (no-board / cluttered) frame never throttles the preview or status."""
        with self.lock:
            fr, intr, board = self.latest_frame, self.intr, self.board
            if fr is None or fr.seq == self.processed_seq:
                return
            self.processed_seq = fr.seq
        if board is None or intr is None:
            return
        det = detect_board(fr.image, board, intr, fast=True)
        if det.ok and fr.frame_id != intr.frame_id:
            det.ok, det.reason = False, 'image frame_id does not match camera_info'
        with self.lock:
            if self.board is not board:
                return  # configuration changed while detecting
            self.live_det, self.live_det_stamp, self.live_det_recv = det, fr.stamp, fr.recv
        if det.ok:
            ts = TransformStamped()
            ts.header.stamp = sec_to_stamp(fr.stamp)
            ts.header.frame_id = fr.frame_id
            ts.child_frame_id = self.board_frame
            ts.transform = T_to_msg(det.T_camera_from_board)
            self.pub_board.publish(ts)

    def publish_preview(self):
        """Newest frame at a fixed rate, overlaid with the latest detection."""
        with self.lock:
            fr, intr, board = self.latest_frame, self.intr, self.board
            det, det_recv = self.live_det, self.live_det_recv
            if fr is None or fr.seq == self.preview_seq:
                return
            self.preview_seq = fr.seq
        out = fr.image
        if det is not None and board is not None and intr is not None:
            lag = fr.recv - det_recv
            if det.ok and lag <= 0.3:
                out = draw_overlay(fr.image, det, board, intr)
            else:
                out = fr.image.copy()
                text = det.reason if not det.ok else 'detecting...'
                cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if self.preview_width and out.shape[1] > self.preview_width:
            h = int(out.shape[0] * self.preview_width / out.shape[1])
            out = cv2.resize(out, (self.preview_width, h), interpolation=cv2.INTER_AREA)
        msg = self.bridge.cv2_to_imgmsg(out, 'bgr8')
        msg.header.stamp = sec_to_stamp(fr.stamp)
        msg.header.frame_id = fr.frame_id
        self.pub_preview.publish(msg)

    # ------------------------------------------------------------------ status
    def publish_status(self):
        now = self.now_s()
        st = CalibrationStatus()
        st.stamp = self.get_clock().now().to_msg()
        rs, rreason = self.robot_live(now)
        with self.lock:
            intr, det, det_stamp, fr = self.intr, self.live_det, self.live_det_stamp, \
                self.latest_frame
            det_recv = self.live_det_recv
            sess = self.session
            st.busy = self.busy
            st.message = self.message
            st.base_frame, st.eef_frame = self.base_frame, self.eef_frame
            st.camera_frame = intr.frame_id if intr else ''
            st.intrinsics_valid = intr is not None
            if self.board is not None:
                st.mode = self.mode
                st.columns, st.rows = self.board.columns, self.board.rows
                st.square_size_mm = float(self.board.square_size_mm)
            if sess is not None:
                st.session_id, st.session_dir = sess.id, sess.dir
                st.config_revision, st.dataset_revision = sess.config_revision, \
                    sess.dataset_revision
                st.num_valid_samples = len(sess.valid_samples())
                st.num_total_samples = len(sess.samples)
            st.result_state = self.result_state()
            last = self.robot_buf[-1] if self.robot_buf else None
            st.robot_mode = last.mode if last else -1
            st.robot_state = last.state if last else -1
            if self.result is not None:
                st.result = self.result['msg']

        # robot
        if rs is not None:
            st.eef_valid = True
            st.eef_stamp = sec_to_stamp(rs.stamp)
            st.eef = T_to_msg(rs.T)
            st.eef_reason = self.tcp_offset_error(rs.offset)
            ts = TransformStamped()
            ts.header.stamp, ts.header.frame_id = st.eef_stamp, self.base_frame
            ts.child_frame_id, ts.transform = self.eef_frame, st.eef
            self.pub_eef.publish(ts)
        else:
            st.eef_reason = rreason
        # board
        img_fresh = fr is not None and now - fr.recv <= self.freshness
        if intr is None:
            st.board_reason = self.intr_reason
        elif fr is None:
            st.board_reason = 'no image received'
        elif not img_fresh:
            st.board_reason = f'image stale ({now - fr.recv:.2f} s)'
        elif self.board is None:
            st.board_reason = 'not configured'
        elif det is None:
            st.board_reason = 'waiting for detection'
        elif not det.ok:
            st.board_reason = det.reason
        elif now - det_recv > self.freshness:
            st.board_reason = f'detection lagging ({now - det_recv:.2f} s)'
        else:
            st.board_valid = True
            st.board_stamp = sec_to_stamp(det_stamp)
            st.board = T_to_msg(det.T_camera_from_board)
            st.board_reason = '; '.join(det.warnings)

        if not st.busy:
            if sess is None:
                st.state = S.UNCONFIGURED
            elif rs is None or intr is None or not img_fresh:
                st.state = S.WAITING_FOR_DATA
            else:
                st.state = S.READY
        else:
            st.state = self.state
        st.can_capture = (st.state == S.READY and st.board_valid and not st.eef_reason)
        st.can_solve = (not st.busy and sess is not None and st.num_valid_samples >= 3)
        st.can_save = (not st.busy and st.result_state == S.RESULT_VALID)
        self.pub_status.publish(st)

    # ------------------------------------------------------------------ services
    def srv_configure(self, req, res):
        board = BoardConfig(int(req.columns), int(req.rows), float(req.square_size_mm))
        err = board.validate()
        if not err and req.mode not in handeye.MODES:
            err = f'mode must be one of {handeye.MODES}'
        if err:
            res.success, res.message = False, err
            return res
        with self.lock:
            if self.busy:
                res.success, res.message = False, 'busy'
                return res
            same = (self.session is not None and self.board == board and self.mode == req.mode
                    and (not req.session_name or req.session_name == self.session.id))
            if same:
                res.success, res.message = True, 'configuration unchanged'
                res.config_revision, res.session_id = self.session.config_revision, \
                    self.session.id
                return res
            self.board, self.mode = board, req.mode
            if self.session is not None and not self.session.samples and not req.session_name:
                # no samples yet: update in place
                self.session.config = self.board_cfg_dict()
                self.session.frames = self.frames_dict()
                self.session.config_revision += 1
                self.session.flush()
                msg = 'configuration updated'
            else:
                # samples never inherit a new configuration -> new session
                try:
                    self.session = Session.create(self.data_root, req.session_name,
                                                  self.board_cfg_dict(), self.frames_dict(),
                                                  self.versions())
                except OSError as e:
                    res.success, res.message = False, f'cannot create session: {e}'
                    return res
                msg = f'new session {self.session.id}'
            self.result, self.result_failed = None, False
            self.live_det = None
            self.processed_seq = -1
            self.message = msg
            res.success, res.message = True, msg
            res.config_revision, res.session_id = self.session.config_revision, self.session.id
        return res

    def srv_capture(self, req, res):
        res.success = False
        with self.lock:
            if self.session is None:
                res.message = 'not configured'
                return res
            if self.busy:
                res.message = 'busy'
                return res
            if req.expected_config_revision and \
                    req.expected_config_revision != self.session.config_revision:
                res.message = 'configuration changed; refresh panel'
                return res
            self.busy, self.state, self.message = True, S.CAPTURING, 'capturing'
            self.capture_frame = None
            self.capture_event.clear()
            self.capture_click = self.now_s()
        try:
            ok, msg, sample = self._capture()
        except Exception as e:  # noqa: BLE001
            ok, msg, sample = False, f'capture error: {e}', None
        with self.lock:
            self.capture_click = None
            self.busy = False
            self.message = msg
            if ok:
                res.success = True
                res.sample_id = sample['sample_id']
                res.reproj_rms_px = sample['reproj_rms_px']
                res.dt_ms = sample['dt_ms']
            res.dataset_revision = self.session.dataset_revision
        res.message = msg
        self.get_logger().info(f'capture: {msg}')
        return res

    def _capture(self):
        if not self.capture_event.wait(self.capture_timeout):
            return False, 'timeout waiting for a new image', None
        with self.lock:
            fr, intr, board, sess = self.capture_frame, self.intr, self.board, self.session
        if intr is None:
            return False, f'intrinsics invalid: {self.intr_reason}', None
        if fr.frame_id != intr.frame_id:
            return False, 'image frame_id does not match camera_info', None
        stored = sess.config.get('intrinsics')
        if stored is not None:
            si = Intrinsics.from_dict(stored)
            if si.frame_id != intr.frame_id or si.width != intr.width or \
                    si.height != intr.height or not np.allclose(si.K, intr.K) or \
                    si.D.shape != intr.D.shape or not np.allclose(si.D, intr.D):
                return False, 'camera intrinsics changed during session; reconfigure', None
        det = detect_board(fr.image, board, intr, fast=True)
        if not det.ok and 'not detected' in det.reason:
            det = detect_board(fr.image, board, intr)  # full-resolution fallback
        if not det.ok:
            return False, f'board: {det.reason}', None
        if det.reproj_rms_px > self.max_reproj_px:
            return False, (f'reprojection RMS {det.reproj_rms_px:.3f} px > '
                           f'{self.max_reproj_px} px'), None

        # wait until the robot buffer covers the static window after the image stamp
        half = self.static_window / 2.0
        deadline = time.monotonic() + self.capture_timeout
        while True:
            with self.lock:
                buf = list(self.robot_buf)
            if buf and buf[-1].stamp >= fr.stamp + half:
                break
            if time.monotonic() > deadline:
                return False, 'timeout waiting for robot_states after image', None
            time.sleep(0.02)
        near = min(buf, key=lambda r: abs(r.stamp - fr.stamp))
        dt = near.stamp - fr.stamp
        if abs(dt) > self.max_dt:
            return False, f'robot/image time difference {dt * 1e3:.1f} ms > ' \
                          f'{self.max_dt * 1e3:.0f} ms', None
        if near.err != 0:
            return False, f'robot error code {near.err}', None
        if near.state == 1 and near.mode != 2:  # in manual mode the static window decides
            return False, 'robot is moving (state RUNNING)', None
        tcp_err = self.tcp_offset_error(near.offset)
        if tcp_err:
            return False, tcp_err, None
        window = [r for r in buf if abs(r.stamp - fr.stamp) <= half]
        if len(window) < 3 or not any(r.stamp <= fr.stamp for r in window) \
                or not any(r.stamp >= fr.stamp for r in window):
            return False, (f'only {len(window)} robot sample(s) around the image in the '
                           f'{self.static_window:.1f} s static window'), None
        dtrans = max(np.linalg.norm(r.T[:3, 3] - near.T[:3, 3]) * 1e3 for r in window)
        drot = max(np.rad2deg(rotation_angle(near.T[:3, :3].T @ r.T[:3, :3])) for r in window)
        if dtrans > self.static_trans_mm or drot > self.static_rot_deg:
            return False, (f'robot not static: {dtrans:.2f} mm / {drot:.3f} deg within '
                           f'{self.static_window:.2f} s'), None

        sample = {
            'image_stamp': fr.stamp, 'image_recv_time': fr.recv,
            'robot_stamp': near.stamp, 'robot_recv_time': near.recv,
            'dt_ms': dt * 1e3,
            'static_window': {'n': len(window), 'max_trans_mm': float(dtrans),
                              'max_rot_deg': float(drot)},
            'robot_raw': {'pose_mm_rad': near.pose, 'tcp_offset_mm_rad': near.offset,
                          'state': near.state, 'mode': near.mode},
            'T_base_from_eef': T_to_list(near.T),
            'T_camera_from_board': T_to_list(det.T_camera_from_board),
            'corners_px': det.corners.tolist(),
            'reproj_rms_px': det.reproj_rms_px,
            'image_frame_id': fr.frame_id,
            'warnings': det.warnings,
        }
        with self.lock:
            if self.session is not sess:
                return False, 'session changed during capture', None
            if 'intrinsics' not in sess.config:
                sess.config['intrinsics'] = intr.to_dict()
                sess.frames['camera_optical'] = intr.frame_id
            sample = sess.add_sample(sample, fr.image)
        n = len(sess.valid_samples())
        warn = f" ({'; '.join(det.warnings)})" if det.warnings else ''
        return True, (f"captured {sample['sample_id']} [{n} valid], reproj "
                      f"{det.reproj_rms_px:.3f} px, dt {dt * 1e3:.1f} ms{warn}"), sample

    def srv_list(self, req, res):
        with self.lock:
            if self.session is None:
                return res
            for s in self.session.samples:
                m = SampleSummary()
                m.sample_id, m.sequence, m.excluded = s['sample_id'], s['sequence'], \
                    s['excluded']
                m.image_stamp = sec_to_stamp(s['image_stamp'])
                m.dt_ms, m.reproj_rms_px = float(s['dt_ms']), float(s['reproj_rms_px'])
                m.warnings = '; '.join(s.get('warnings', []))
                res.samples.append(m)
            res.dataset_revision = self.session.dataset_revision
        return res

    def srv_delete(self, req, res):
        with self.lock:
            if self.session is None or self.busy:
                res.success, res.message = False, 'not configured' if self.session is None \
                    else 'busy'
                return res
            target, err = self.session.exclude(req.sample_id or 'last')
            res.dataset_revision = self.session.dataset_revision
            if target is None:
                res.success, res.message = False, err
            else:
                res.success, res.sample_id = True, target['sample_id']
                res.message = f"excluded {target['sample_id']}"
            self.message = res.message
        return res

    def srv_save(self, req, res):
        res.success = False
        with self.lock:
            if self.busy:
                res.message = 'busy'
                return res
            if self.result_state() != S.RESULT_VALID:
                res.message = 'no valid result for the current dataset; solve first'
                return res
            r, sess = self.result, self.session
            X = r['X']
            q = matrix_to_quat_xyzw(X[:3, :3])
            if not np.allclose(quat_xyzw_to_matrix(q), X[:3, :3], atol=1e-9):
                res.message = 'internal error: quaternion/matrix mismatch'
                return res
            extr = {
                'mode': r['mode'],
                'parent_frame': r['parent_frame'],
                'child_frame': r['child_frame'],
                'transform_direction': 'T_parent_from_child (maps child-frame points into '
                                       'parent frame)',
                'units': {'translation': 'm', 'rotation': 'unit quaternion / matrix'},
                'translation': X[:3, 3].tolist(),
                'quaternion_xyzw': q.tolist(),
                'rotation_matrix': X[:3, :3].tolist(),
                'matrix_4x4': X.tolist(),
                'note': 'child is the camera OPTICAL frame, not camera_link',
            }
            report = dict(r['report'])
            try:
                path = sess.save_result(req.name, req.overwrite, extr, report)
            except (OSError, ValueError) as e:
                res.message = str(e)
                return res
            res.success, res.path, res.message = True, path, f'saved {path}'
            self.message = res.message
        return res

    def srv_load(self, req, res):
        res.success = False
        with self.lock:
            if self.busy:
                res.message = 'busy'
                return res
        path = req.session_dir
        if not os.path.isabs(path):
            path = os.path.join(self.data_root, path)
        try:
            sess = Session.load(path)
            c = sess.config
            board = BoardConfig(int(c['columns']), int(c['rows']), float(c['square_size_mm']))
            if board.validate() or c['mode'] not in handeye.MODES:
                raise ValueError('invalid configuration in session')
        except Exception as e:  # noqa: BLE001
            res.message = f'load failed: {e}'
            return res
        with self.lock:
            self.session, self.board, self.mode = sess, board, c['mode']
            self.result, self.result_failed = None, False
            self.processed_seq = -1
            res.success = True
            res.session_id, res.config_revision, res.dataset_revision = \
                sess.id, sess.config_revision, sess.dataset_revision
            res.message = f'loaded {sess.id} ({len(sess.valid_samples())} valid samples)'
            if sess.frames.get('eef') != self.eef_frame:
                res.message += f"; WARNING session eef frame {sess.frames.get('eef')}"
            self.message = res.message
        return res

    # ------------------------------------------------------------------ robot mode
    def _call(self, client, req, timeout=3.0):
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f'{client.srv_name} not available')
        ev = threading.Event()
        fut = client.call_async(req)
        fut.add_done_callback(lambda _: ev.set())
        if not ev.wait(timeout):
            raise RuntimeError(f'{client.srv_name} timed out')
        res = fut.result()
        if res.ret != 0:
            raise RuntimeError(f'{client.srv_name} returned {res.ret}: {res.message}')

    def set_robot_mode(self, manual):
        """manual=True: mode 2 (drag by hand); False: mode 0 (position, holds pose)."""
        self._call(self.cli_enable, SetInt16ById.Request(id=8, data=1))
        self._call(self.cli_mode, SetInt16.Request(data=2 if manual else 0))
        self._call(self.cli_state, SetInt16.Request(data=0))
        return 'robot in MANUAL mode (drag to pose, release, then Capture)' if manual \
            else 'robot in POSITION mode (holding pose)'

    def srv_manual(self, req, res):
        try:
            res.message = self.set_robot_mode(bool(req.data))
            res.success = True
        except RuntimeError as e:
            res.success, res.message = False, f'mode switch failed: {e}'
        with self.lock:
            self.message = res.message
        self.get_logger().info(res.message)
        return res

    def manual_on_start_tick(self):
        if not self.manual_on_start or self.manual_start_done:
            return
        rs, _ = self.robot_live(self.now_s())
        if rs is None:
            return  # wait for a healthy robot (err == 0)
        self.manual_start_done = True
        if rs.mode == 2:
            return
        try:
            msg = self.set_robot_mode(True)
        except RuntimeError as e:
            msg = f'could not enter manual mode on start: {e}'
        with self.lock:
            self.message = msg
        self.get_logger().info(msg)

    # ------------------------------------------------------------------ solve
    def goal_solve(self, goal):
        with self.lock:
            if self.busy or self.session is None:
                return GoalResponse.REJECT
            self.busy, self.state, self.message = True, S.SOLVING, 'solving'
        return GoalResponse.ACCEPT

    def exec_solve(self, gh):
        result = Solve.Result()
        try:
            ok, msg = self._solve(gh, result)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f'solve error: {e}'
        result.success, result.message = ok, msg
        with self.lock:
            self.busy = False
            self.message = msg
            if not ok and not gh.is_cancel_requested:
                self.result, self.result_failed = None, True
        self.get_logger().info(f'solve: {msg}')
        if gh.is_cancel_requested:
            gh.canceled()
        elif ok:
            gh.succeed()
        else:
            gh.abort()
        return result

    def camera_link_parent(self):
        """Current parent of camera_link in the TF tree, or None."""
        try:
            frames = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
        except yaml.YAMLError:
            return None
        entry = frames.get(self.camera_link_frame)
        return entry.get('parent') if isinstance(entry, dict) else None

    def publish_camera_tf(self, X, parent, optical):
        """Publish T_parent_from_camera_link = X @ T_optical_from_camera_link on /tf_static.

        X is T_parent_from_optical. camera_link -> optical is owned by the RealSense driver,
        so we attach the result at camera_link instead of giving the optical frame a 2nd parent.
        """
        if not optical:
            return 'TF not published: unknown camera optical frame'
        try:
            tf = self.tf_buffer.lookup_transform(optical, self.camera_link_frame, Time())
        except TransformException as e:
            return f'TF not published: cannot look up {self.camera_link_frame} -> {optical} ({e})'
        t, q = tf.transform.translation, tf.transform.rotation
        T_opt_link = make_T(quat_xyzw_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])
        existing = self.camera_link_parent()
        if existing and existing != parent and (self.tf_published is None
                                                or self.tf_published[0] != existing):
            return (f'TF not published: {self.camera_link_frame} already has parent '
                    f'"{existing}" from another publisher')
        T = X @ T_opt_link
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = parent
        ts.child_frame_id = self.camera_link_frame
        ts.transform = T_to_msg(T)
        self.tf_static.sendTransform(ts)
        self.tf_published = (parent, self.camera_link_frame)
        return f'TF published: {parent} -> {self.camera_link_frame} (/tf_static)'

    def _feedback(self, gh, stage, progress):
        fb = Solve.Feedback()
        fb.stage, fb.progress = stage, float(progress)
        gh.publish_feedback(fb)

    def _solve(self, gh, result):
        method = (gh.request.method or 'PARK').upper()
        with self.lock:
            sess = self.session
            if gh.request.expected_dataset_revision and \
                    gh.request.expected_dataset_revision != sess.dataset_revision:
                return False, 'dataset changed; refresh and retry'
            snap = [dict(s) for s in sess.valid_samples()]
            mode = sess.config['mode']
            cfg_rev, ds_rev, sid = sess.config_revision, sess.dataset_revision, sess.id
        self._feedback(gh, 'checking motion', 0.1)
        B = [list_to_T(s['T_base_from_eef']) for s in snap]
        C = [list_to_T(s['T_camera_from_board']) for s in snap]
        mc = handeye.check_motion(B, min_rel_rot_deg=self.min_rel_rot_deg,
                                  min_axis_spread_deg=self.min_axis_spread_deg)
        if not mc.ok:
            return False, 'insufficient motion: ' + '; '.join(mc.errors)
        if gh.is_cancel_requested:
            return False, 'canceled'
        self._feedback(gh, f'calibrateHandEye {method}', 0.4)
        he = handeye.solve(mode, B, C, method)
        if not he.ok:
            return False, he.message
        if gh.is_cancel_requested:
            return False, 'canceled'
        self._feedback(gh, 'quality report', 0.8)

        cam = sess.frames.get('camera_optical', '')
        parent = self.eef_frame if mode == 'eye_in_hand' else self.base_frame
        reproj = [float(s['reproj_rms_px']) for s in snap]
        rr = CalibrationResult()
        rr.mode, rr.parent_frame, rr.child_frame = mode, parent, cam
        rr.transform = T_to_msg(he.X)
        rr.method, rr.config_revision, rr.dataset_revision = method, cfg_rev, ds_rev
        rr.sample_ids = [s['sample_id'] for s in snap]
        rr.reproj_rms_px = float(np.sqrt(np.mean(np.square(reproj))))
        rr.consistency_trans_rms_mm, rr.consistency_trans_max_mm = he.trans_rms_mm, \
            he.trans_max_mm
        rr.consistency_rot_rms_deg, rr.consistency_rot_max_deg = he.rot_rms_deg, he.rot_max_deg
        fixed_name = 'T_base_from_board (Y)' if mode == 'eye_in_hand' else \
            'T_eef_from_board (Z)'
        rr.report = (f'{len(snap)} samples; reproj RMS {rr.reproj_rms_px:.3f} px; '
                     f'{fixed_name} consistency: {he.trans_rms_mm:.2f} mm RMS / '
                     f'{he.trans_max_mm:.2f} max, {he.rot_rms_deg:.3f} deg RMS / '
                     f'{he.rot_max_deg:.3f} max')
        cmp = handeye.compare_methods(mode, B, C, he.X)
        bad = [k for k, v in cmp.items() if v['ok'] and
               (v['diff_trans_mm'] > self.method_warn_mm or v['diff_rot_deg'] > self.method_warn_deg)]
        warnings = list(mc.warnings)
        if bad:
            warnings.append(f'methods {bad} differ from {method} by > {self.method_warn_mm} mm / '
                            f'{self.method_warn_deg} deg; check data/method')
        if warnings:
            rr.report += ' | ' + '; '.join(warnings)
        report = {
            'method': method, 'opencv_version': cv2.__version__,
            'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'session_id': sid, 'config_revision': cfg_rev, 'dataset_revision': ds_rev,
            'config': {k: v for k, v in sess.config.items()},
            'frames': dict(sess.frames),
            'sample_ids': rr.sample_ids,
            'reprojection_rms_px': {'overall': rr.reproj_rms_px,
                                    'per_sample': dict(zip(rr.sample_ids, reproj))},
            'hand_eye_consistency': {
                'quantity': fixed_name,
                'method': 'translation distance to mean translation; geodesic angle to '
                          'chordal L2 mean rotation on SO(3)',
                'reference_mean': he.fixed_mean.tolist(),
                'trans_mm': {'rms': he.trans_rms_mm, 'max': he.trans_max_mm,
                             'per_sample': dict(zip(rr.sample_ids, he.per_sample_trans_mm))},
                'rot_deg': {'rms': he.rot_rms_deg, 'max': he.rot_max_deg,
                            'per_sample': dict(zip(rr.sample_ids, he.per_sample_rot_deg))},
                'note': 'internal consistency only; not the absolute extrinsic error',
            },
            'motion_check': {'max_relative_rotation_deg': mc.max_rel_rot_deg,
                             'max_axis_spread_deg': mc.max_axis_spread_deg,
                             'warnings': mc.warnings},
            'method_comparison': {'reference': method, 'others': cmp,
                                  'note': 'agreement is not proof of absolute accuracy'},
            'warnings': warnings,
            'thresholds': {'max_pair_dt_ms': self.max_dt * 1e3,
                           'static_window_s': self.static_window,
                           'static_trans_mm': self.static_trans_mm,
                           'static_rot_deg': self.static_rot_deg,
                           'max_reproj_px': self.max_reproj_px,
                           'min_rel_rot_deg': self.min_rel_rot_deg,
                           'min_axis_spread_deg': self.min_axis_spread_deg},
        }
        with self.lock:
            if self.session is not sess:
                return False, 'session changed during solve'
            self.result = {'X': he.X, 'mode': mode, 'parent_frame': parent, 'child_frame': cam,
                           'session_id': sid, 'config_revision': cfg_rev,
                           'dataset_revision': ds_rev, 'msg': rr, 'report': report}
            self.result_failed = False
        tf_msg = self.publish_camera_tf(he.X, parent, cam) if self.publish_tf else 'TF publishing disabled'
        rr.report += f' | {tf_msg}'
        report['tf'] = tf_msg
        result.result = rr
        self._feedback(gh, 'done', 1.0)
        return True, 'solved: ' + rr.report


def main():
    rclpy.init()
    node = CalibrationNode()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
