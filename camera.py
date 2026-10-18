import cv2
import numpy as np
import pyrealsense2 as rs
from cv2 import Rodrigues

def get_intrinsics(pth = 'intrinsics.npz'):
    calibration_data = np.load(pth)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']
    return camera_matrix, dist_coeffs

class Camera:
    def __init__(self):
        self.camera_matrix, self.dist_coeffs = get_intrinsics()
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.aruco_init()
        # self.item_size_dict = {
        #                        '0':np.array([180,220,110]),
        #                        '1':np.array([130,220,160]),
        #                        '2':np.array([120,180,200]),
        #                        '5':np.array([190,240,80]),
        #                        '6':np.array([170,170,150]),
        #                        '7':np.array([190,190,110]),
        #                        '9':np.array([190,240,130]),
        #                        }
        self.item_size_dict = {
                               '0':np.array([220,180,110]),
                               '1':np.array([230,130,160]),
                               '2':np.array([180,130,200]),
                               '4':np.array([120,120,150]),
                               '5':np.array([240,190,80]),
                               '6':np.array([170,170,150]),
                               '7':np.array([190,190,110]),
                               '9':np.array([240,190,130]),
                               }

    def aruco_init(self):
        # Define the ArUco dictionary to be used
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Set marker length in meters (e.g., 60mm = 0.06m)
        self.marker_length = 0.065

    def get_color(self):
        while True:
            # Capture frames from RealSense
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            else:
                img_arr = np.asanyarray(color_frame.get_data())
                return img_arr
    
    def detect_markers(self, img_arr, draw=True):
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None:
            # Estimate pose of markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(img_arr, corners, ids)
            for i, marker_id in enumerate(ids):
                # Draw axis for each marker
                cv2.drawFrameAxes(img_arr, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length / 2)

                # Print marker pose
                # print(f"Marker ID: {marker_id[0]}")
                # print(f"Rotation Vector: {rvecs[i].flatten()}")
                # # print(f"Translation Vector: {R_eef_cam.dot(tvecs[i].T).flatten()}")
                # print(f"Translation Vector: {tvecs[i].flatten()}")
            return ids, corners, rvecs, tvecs, img_arr
        else:
            return ids, None, None, None, img_arr

    def select_marker(self, ids, tvecs):
        #find the marker that is closet along z axis
        idx = np.array(tvecs).reshape(-1, 3)[:,-1].argmin()
        return idx
    
    def get_H_cam_target(self, show=True):
        img_arr = self.get_color()
        ids, corners, rvecs, tvecs, img_arr = self.detect_markers(img_arr)
        if ids is not None:
            idx = self.select_marker(ids=ids, tvecs=tvecs)
            if show:
                cv2.aruco.drawDetectedMarkers(img_arr, [corners[idx]], np.array([ids[idx]]), borderColor=(0,0,255))
                # Draw axis for each marker
                cv2.drawFrameAxes(img_arr, self.camera_matrix, 
                                self.dist_coeffs, 
                                rvecs[idx], 
                                tvecs[idx], 
                                self.marker_length / 2)
            R = Rodrigues(rvecs[idx])
            H = np.eye(4)
            H[:3,:3] = R[0]
            H[:3,-1] = np.array(tvecs[idx]).squeeze()*1000
            return H, img_arr, ids[idx].item()
        else:
            print("markers were not detected.")
            return None, img_arr, None

    def show(self):
        try:
            while True:
                H, img_arr = self.get_H_cam_target()
                print('H matrix: {}'.format(H))
                cv2.imshow('ArUco Detection', img_arr)
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
    
    def end(self):
        self.pipeline.stop()

if __name__ == "__main__":
    cam = Camera()
    cam.show()