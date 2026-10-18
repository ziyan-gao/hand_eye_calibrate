import cv2
import numpy as np
import glob
import os
import time
from xarm.wrapper import XArmAPI
import pickle
import os
import pyrealsense2 as rs
import numpy as np
import cv2

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


def get_eef_H(arm):
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
    return (H, [x, y,z, roll, pitch, yaw])


def main():
    # Initialize XARM
    robot_ip = '192.168.1.232'  # Replace with your robot's IP address
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_mode(0)  # 0: Position control
    arm.set_state(0)
    time.sleep(1)
    # Create output directory for saved frames
    save_folder = "chessboard_extrinsics_new"
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
    eef_H_list = {}

    try:
        while True:
            # Wait for the next frame and retrieve the color frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert color frame to a NumPy array
            color_image = np.asanyarray(color_frame.get_data())

            # Display the RGB stream
            cv2.imshow('RealSense - RGB Only', color_image)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            # Press 'i' to save the current frame
            if key == ord('i'):
                image_count += 1
                filename = os.path.join(save_folder, f"{image_count}.png")
                cv2.imwrite(filename, color_image)
                print(f"Saved: {filename}")
                H = get_eef_H(arm=arm)
                eef_H_list[filename] = H

            # Optional: press 'q' or ESC (27) to quit
            if key == ord('q') or key == 27:
                print("Exiting...")
                break
    except Exception as e:
        print(e)

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        # Disconnect from robot
        arm.disconnect()

    with open(os.path.join(save_folder, 'eef_homogeneous.pickle'), 'wb') as f:
        pickle.dump(eef_H_list, f)


if __name__ == '__main__':
    main()