import cv2
import numpy as np
import pyrealsense2 as rs
from cv2 import Rodrigues
def get_intrinsics(pth = 'intrinsics.npz'):
    calibration_data = np.load(pth)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']
    return camera_matrix, dist_coeffs

def get_extrinsics(pth = 'extrinsics.npz'):
    cam2eef = np.load(pth)
    R = cam2eef['R']
    tvec = cam2eef['T']
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, -1] = tvec.squeeze()
    return H

def main():
    # Define the ArUco dictionary to be used
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # Set marker length in meters (e.g., 60mm = 0.06m)
    marker_length = 0.0645

    camera_matrix, dist_coeffs = get_intrinsics()
    H = get_extrinsics()
    R_eef_cam = H[:3, :3]

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            # Capture frames from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert RealSense frame to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

            if ids is not None:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Estimate pose of markers
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

                for i, marker_id in enumerate(ids):
                    # Draw axis for each marker
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length / 2)

                    # Print marker pose
                    print(f"Marker ID: {marker_id[0]}")
                    print(f"Rotation Vector: {rvecs[i].flatten()}")
                    # print(f"Translation Vector: {R_eef_cam.dot(tvecs[i].T).flatten()}")
                    print(f"Translation Vector: {tvecs[i].flatten()}")

            # Show frame with detected markers
            cv2.imshow('ArUco Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Stop RealSense pipeline and close OpenCV window
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()