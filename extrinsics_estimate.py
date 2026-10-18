import cv2
import numpy as np
import pickle
import os
from cv2 import calibrateHandEye, CALIB_HAND_EYE_TSAI, Rodrigues
import matplotlib.pyplot as plt

def get_intrinsics(pth = 'chessboard/calibration_result.npz'):
    calibration_data = np.load(pth)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']
    return camera_matrix, dist_coeffs

def save_calibration(R_handeye, T_handeye, save_path):
    """
    Save the camera calibration results to a file.

    Parameters:
    - camera_matrix: Camera intrinsic matrix.
    - dist_coeffs: Distortion coefficients.
    - save_path: Directory to save the calibration file.
    """
    calibration_file = os.path.join(save_path, 'cam2eef.npz')
    np.savez(calibration_file, R=R_handeye, T=T_handeye)
    print(f"\nCalibration results saved to {calibration_file}")

def extrinsics_estimate(camera_matrix, dist_coeffs, 
                        data_pth='chessboard_extrinsics/eef_homogeneous.pickle'):
    # Step 1: Load End-Effector Poses
    with open(data_pth, 'rb') as handle:
        eef_poses = pickle.load(handle)

    # Extract and sort image paths for consistency
    image_paths = sorted(eef_poses.keys())

    # Step 2: Load and Process Images
    chessboard_size = (10, 7)
    square_size = 33  # in millimeters

    # Prepare object points
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    valid_image_paths = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # Refine corner locations
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            obj_points.append(objp)
            valid_image_paths.append(img_path)
            # Optional: Visualize detected corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(2000)
        else:
            print(f"Chessboard corners not found in image: {img_path}")

    # cv2.destroyAllWindows()
    print(f"Number of valid calibration images: {len(valid_image_paths)}")

    # Step 3: Extract and Convert Robot Poses
    R_gripper2base = []
    t_gripper2base = []

    for img_path in valid_image_paths:
        H, _ = eef_poses[img_path]
        R = H[:3,:3]
        T = H[:3,-1]
        R_gripper2base.append(R)
        t_gripper2base.append(T)


    R_target2cam = []
    T_target2cam = []

    for i in range(len(obj_points)):
        objp_current = obj_points[i]
        imgp_current = img_points[i]
        ret, rvec, tvec = cv2.solvePnP(objp_current, imgp_current, camera_matrix, dist_coeffs)
        if ret:
            R_cam, _ = Rodrigues(rvec)  # Convert rotation vector to rotation matrix
            T_cam = tvec.flatten()
            R_target2cam.append(R_cam)
            T_target2cam.append(T_cam)
        else:
            print(f"solvePnP failed for image: {valid_image_paths[i]}")


    # Perform calibration
    R_hand_eye, T_hand_eye = calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        T_target2cam,
        method=CALIB_HAND_EYE_TSAI
    )

    print("Hand-Eye Calibration Results:")
    print("Rotation Matrix (R_hand_eye):\n", R_hand_eye)
    print("Translation Vector (T_hand_eye):\n", T_hand_eye)
    return R_hand_eye, T_hand_eye

if __name__ == '__main__':
    intrinsics_data_pth = 'chessboard/calibration_result.npz'
    extrinsics_data_pth = 'chessboard_extrinsics_new/eef_homogeneous.pickle'
    camera_matrix, dist_coeffs = get_intrinsics(intrinsics_data_pth)
    R_hand_eye, T_hand_eye = extrinsics_estimate(camera_matrix, dist_coeffs, extrinsics_data_pth)
    save_calibration(R_hand_eye, T_hand_eye, save_path='chessboard_extrinsics_new')