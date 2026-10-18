import cv2
import numpy as np
import glob
import os

def calibrate_camera(chessboard_folder, square_size_mm, chessboard_size=(9, 6)):
    """
    Calibrate the camera using chessboard images.

    Parameters:
    - chessboard_folder: Path to the folder containing chessboard images.
    - square_size_mm: Size of a square in millimeters.
    - chessboard_size: Number of internal corners per a chessboard row and column (columns, rows).

    Returns:
    - ret: RMS re-projection error.
    - camera_matrix: Camera intrinsic matrix.
    - dist_coeffs: Distortion coefficients.
    - rvecs: Rotation vectors.
    - tvecs: Translation vectors.
    """
    # Termination criteria for cornerSubPix
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    # Prepare object points based on the actual chessboard dimensions
    objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm  # Scale by square size to get real-world coordinates

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Get list of images
    images = glob.glob(os.path.join(chessboard_folder, '*.png'))
    if not images:
        print(f"No images found in {chessboard_folder}. Ensure the path is correct and images are in PNG format.")
        return

    # Process each image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image {fname}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)

            # Refine corner locations to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners (optional)
            img_drawn = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners2, ret)
            cv2.imshow('Chessboard Detection', img_drawn)
            cv2.waitKey(5000)  # Display each image for 100ms
        else:
            print(f"Chessboard corners not found in image {fname}. Skipping.")

    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("No chessboard corners were detected in any image. Calibration cannot proceed.")
        return

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate re-projection error
    mean_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        mean_error += error**2
        total_points += len(objpoints[i])

    mean_error = np.sqrt(mean_error / total_points)
    print(f"Calibration RMS Re-projection Error: {ret}")
    print(f"Mean Re-projection Error: {mean_error}")

    print("\nCamera Matrix:")
    print(camera_matrix)

    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())

    # Optionally, save the calibration results
    save_calibration(camera_matrix, dist_coeffs, chessboard_folder)

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

def save_calibration(camera_matrix, dist_coeffs, save_path):
    """
    Save the camera calibration results to a file.

    Parameters:
    - camera_matrix: Camera intrinsic matrix.
    - dist_coeffs: Distortion coefficients.
    - save_path: Directory to save the calibration file.
    """
    calibration_file = os.path.join(save_path, 'calibration_result.npz')
    np.savez(calibration_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\nCalibration results saved to {calibration_file}")

if __name__ == "__main__":
    # Parameters
    chessboard_folder = "chessboard"  # Folder containing saved chessboard images
    square_size_mm = 33  # Size of a square in millimeters
    chessboard_size = (10, 7)  # Number of internal corners per chessboard row and column

    calibrate_camera(chessboard_folder, square_size_mm, chessboard_size)