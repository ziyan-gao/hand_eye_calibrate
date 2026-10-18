import os
import sys
import time
import csv
from datetime import datetime
import math
import numpy as np
from xarm.wrapper import XArmAPI

from cv2 import Rodrigues
import matplotlib.pyplot as plt

# import transformations as tf
import time
from functools import wraps

from Execute_pushing_profile import rotationMatrixToEulerAngles

def delayBeforeExec(dt, count=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            t0 = time.time()
            time.sleep(dt)
            ret = func(*args, **kw)
            print(
                "%s() is delayed before execution.\n"
                % (func.__name__)
            )
            if count:
                print(
                    "total time cost: %s s\n"
                    % (str(np.round(time.time() - t0, 3)))
                )
            return ret

        return wrapper

    return decorator


def delayAfterExec(dt, count=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            t0 = time.time()
            ret = func(*args, **kw)
            time.sleep(dt)
            print(
                "%s() is delayed after execution.\n"
                % (func.__name__)
            )
            if count:
                print(
                    "total time cost: %s s\n"
                    % (str(np.round(time.time() - t0, 3)))
                )
            return ret

        return wrapper

    return decorator


class Arm:
    def __init__(self):
        ip = '192.168.1.232'
        # Initialize xArm
        self.arm = XArmAPI(ip)
        # Enable motion and set initial state
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.H_eef_cam = self.get_H_eef_cam()
        self.obs_pose = np.array([300.0, -20.0, 400, -180.00001985, -0.0, 0.0])
        # self.pending_left = np.array([400,-300,600,-180,0,0])
        # self.pending_left = np.array([400,300,600,-180,0,0])
    
    @property
    def pose(self):
        res, pose = self.arm.get_position(is_radian=False)
        return pose
    
    @property
    def position(self):
        res, pose = self.arm.get_position(is_radian=False)
        return pose[:3]

    @property
    def euler(self):
        res, pose = self.arm.get_position(is_radian=False)
        return pose[3:]

    @property
    def H_base_eef(self):
        # Assuming the pose includes position and orientation as Euler angles
        x, y, z, roll, pitch, yaw = self.pose[:6]

        # Convert Euler angles to a rotation matrix
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)

        # Create the translation vector
        T = np.array([[x], [y], [z]])

        # Assemble the homogeneous transformation matrix
        H = np.eye(4)  # Create a 4x4 identity matrix
        H[:3, :3] = R  # Insert the rotation matrix
        H[:3, 3] = T.flatten()  # Insert the translation vector
        return H

    def set_joint_positions(self, 
                            angle=[-66.08177286136072, -30.770616927932743, -73.86965017565764, 
                                   0.06295165911493239, -42.990590334272596, -66.21328132364015],
                            speed=200):
        
        self.arm.set_servo_angle(angle=angle, speed=speed/10, wait=True)

    @delayAfterExec(0.3)#>0.3
    def setOrient(self, angle, relative=True, speed=200):
        js = self.arm.get_joint_states()[1][0][:6]
        currentOri = js[-1]
        if relative:
            absOri = currentOri - angle
        else:
            absOri = -angle
        self.arm.set_servo_angle(angle=[*js[:5], absOri], speed=speed, wait=True)

    @delayAfterExec(0.3)#>0.6
    def set_position(self, pose, is_radian=False, relative=False, speed=30):
        if len(pose) == 6:
            # self.setOrient(0, False)
            ret = self.arm.set_position(*pose, is_radian=is_radian, wait=True, speed=speed, relative=relative)
        elif len(pose) == 3:
            ret = self.arm.set_position(x = pose[0], y=pose[1], z=pose[2], is_radian=is_radian,speed=speed, wait=True, relative=relative)
            # self.setOrient(0, False)
        elif len(pose) == 2:
            ret = self.arm.set_position(x = pose[0], y=pose[1], is_radian=is_radian,speed=speed, wait=True, relative=relative)
            # self.setOrient(0, False)
        time.sleep(0.3)
    
    @delayAfterExec(0.3)#>0.6
    def zmove(self, z, relative=False, speed=50):
        current = self.pose
        xy = current[:2]
        orient = current[3:]
        if relative:
            abs_z = current[2] + z
        else:
            abs_z = z
        self.arm.set_position(
            *xy, abs_z, *orient, relative=False, wait=True, speed=speed
        )

    def get_H_eef_cam(self, pth = 'extrinsics.npz'):
        cam2eef = np.load(pth)
        R = cam2eef['R']
        tvec = cam2eef['T']
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, -1] = tvec.squeeze()
        return H

    def degrees_to_radians(self, degrees):
        return degrees * np.pi / 180.0

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Convert Euler angles to a rotation matrix.
        Roll (X-axis), Pitch (Y-axis), Yaw (Z-axis)
        """
        # Convert degrees to radians
        roll_rad = self.degrees_to_radians(roll)
        pitch_rad = self.degrees_to_radians(pitch)
        yaw_rad = self.degrees_to_radians(yaw)
        
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

    def gohome(self):
        self.arm.reset(wait=True, speed=50)

    def goToObsPos(self):
        self.set_position(self.obs_pose, speed=300)
    
    def end(self):
        self.close()
        self.arm.disconnect()

    @staticmethod
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


if __name__ == "__main__":
    arm = Arm()
    arm.goToObsPos()
    time.sleep(1)
    arm.gohome()
    #set desired push force
    # push_force = 10
    # pxyz = [400,0,250]
    # arm.set_position(pxyz)
    # try:
    #     arm.exert_push_force(push_force=push_force)
    # except Exception as e:
    #     print(e)
    
    # arm.end()

