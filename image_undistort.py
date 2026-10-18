import os
import pyrealsense2 as rs
import numpy as np
import cv2

# Load calibration data
calibration_data = np.load('chessboard/calibration_result.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Create output directory for saved frames
save_folder = "chessboard"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Configure color stream only
pipeline = rs.pipeline()
config = rs.config()

# Check for RGB camera availability
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires a RealSense Depth camera with a Color (RGB) sensor.")
    exit(0)

# Enable only the color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Counter for saved images
image_count = 0

try:
    while True:
        # Wait for the next frame and retrieve the color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Assuming camera_matrix and dist_coeffs are already defined from calibration
        h, w = color_image.shape[:2]

        # alpha=0 retains all valid pixels, potentially cropping the image
        # alpha=1 retains all original pixels, potentially adding black borders
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0)
        # Undistort the image
        undistorted_image = cv2.undistort(color_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Display the undistorted RGB stream
        cv2.imshow('RealSense - Undistorted RGB', undistorted_image)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # Press 'i' to save the current undistorted frame
        if key == ord('i'):
            image_count += 1
            filename = os.path.join(save_folder, f"undistorted_{image_count}.png")
            cv2.imwrite(filename, undistorted_image)
            print(f"Saved: {filename}")

        # Optional: press 'q' or ESC (27) to quit
        if key == ord('q') or key == 27:
            print("Exiting...")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()