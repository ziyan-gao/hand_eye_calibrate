import cv2
import numpy as np
import glob
import os
import time
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
from cv2 import Rodrigues

def get_intrinsics(pth = 'intrinsics.npz'):
    calibration_data = np.load(pth)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']
    return camera_matrix, dist_coeffs

def degrees_to_radians(degrees):
    return degrees * np.pi / 180.0

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a rotation matrix.
    Roll (X-axis), Pitch (Y-axis), Yaw (Z-axis)
    """
    # Convert degrees to radians
    roll_rad = degrees_to_radians(roll)
    pitch_rad = degrees_to_radians(pitch)
    yaw_rad = degrees_to_radians(yaw)
    
    # Rotation matrices around the X, Y, and Z axes
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def get_H_base_eef(arm):
    # Get the current pose of the end-effector
    res, pose = arm.get_position()

    # Assuming the pose includes position and orientation as Euler angles
    x, y, z, roll, pitch, yaw = pose[:6]

    # Convert Euler angles to a rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # Create the translation vector
    T = np.array([[x], [y], [z]])

    # Assemble the homogeneous transformation matrix
    H = np.eye(4)  # Create a 4x4 identity matrix
    H[:3, :3] = R  # Insert the rotation matrix
    H[:3, 3] = T.flatten()  # Insert the translation vector

    print("Homogeneous Transformation Matrix:\n", H)
    return H

def get_H_eef_cam(pth = 'extrinsics.npz'):
    cam2eef = np.load(pth)
    R = cam2eef['R']
    tvec = cam2eef['T']
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, -1] = tvec.squeeze()
    return H


def get_H_cam_target(frame):#rgb array
    # Define the ArUco dictionary to be used
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # Set marker length in meters (e.g., 60mm = 0.06m)
    marker_length = 0.06

    camera_matrix, dist_coeffs = get_intrinsics()

    # # Initialize RealSense pipeline
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # pipeline.start(config)

    # try:
    #     while True:
    #         # Capture frames from RealSense
    #         frames = pipeline.wait_for_frames()
    #         color_frame = frames.get_color_frame()
    #         if not color_frame:
    #             continue

    # Convert RealSense frame to numpy array
    # frame = np.asanyarray(color_frame.get_data())

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of markers
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
        R = Rodrigues(rvecs)[0]
        H = np.eye(4)
        H[:3,:3] = R
        H[:3,-1] = np.array(tvecs).squeeze()*1000
        return H
    else:
        return 

import time

def rotationMatrixToEulerAngles(R):
    """
    Decompose the rotation matrix R (shape: (3,3)) into roll, pitch, yaw angles (in radians),
    assuming the rotation sequence R = Rz(yaw)*Ry(pitch)*Rx(roll).

    Returns (roll, pitch, yaw) in radians.
    """

    # Safeguard: if using floating computations, you may get slight numerical drift outside [-1, +1].
    # We clamp the value for asin() to avoid errors if e.g. -1.000000002 occurs.
    def clamp(v):
        return max(min(v, 1.0), -1.0)

    # Extract angles
    # R[2,0] = -sin(pitch)
    pitch = -np.arcsin(clamp(R[2,0]))

    # roll = atan2(R[2,1], R[2,2])
    roll  = np.arctan2(R[2,1], R[2,2])

    # yaw = atan2(R[1,0], R[0,0])
    yaw   = np.arctan2(R[1,0], R[0,0])

    return [roll, pitch, yaw]

def main():
    # Initialize XARM
    robot_ip = '192.168.1.232'  # Replace with your robot's IP address
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_mode(0)  # 0: Position control
    arm.set_state(0)
    time.sleep(1)

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    t0 = time.time()

    try:
        while True:
            # Capture frames from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RealSense frame to numpy array
            frame = np.asanyarray(color_frame.get_data())

            if time.time() - t0 > 3:
                pipeline.stop()
                break
    except Exception as e:
        print(e)
    
    H_cam_target = get_H_cam_target(frame)
    H_base_eef = get_H_base_eef(arm=arm)
    H_eef_cam = get_H_eef_cam()
    #z offset
    H_base_target = H_base_eef.dot(H_eef_cam).dot(H_cam_target)
    Rx_180 = np.array(
        [
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1]
        ]
    )

    Tz_250 = np.array(
        [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,-200],
            [0,0,0,1]
        ]
    )
    H_base_target = H_base_target.dot(Rx_180).dot(Tz_250)

    eulers = rotationMatrixToEulerAngles(H_base_target[:3,:3])
    pose = H_base_target[:3,-1].tolist() + eulers
    
    arm.set_position(*pose, is_radian=True, wait=True,speed=30)


if __name__ == '__main__':
    main()

