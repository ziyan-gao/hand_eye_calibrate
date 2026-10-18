"""Robot driver + RealSense + calibration backend + RViz2 with the calibration panel."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lc = LaunchConfiguration
    share = FindPackageShare('camera_extrinsic_calibration_bringup')
    args = [
        DeclareLaunchArgument('robot_ip', default_value='192.168.1.234'),
        # 'normal' (5 Hz) carries err and TCP offset; 'dev' (100 Hz) does NOT (both read 0),
        # which would defeat the TCP-offset check. Smooth RViz comes from joint_states_rate.
        DeclareLaunchArgument('report_type', default_value='normal'),
        DeclareLaunchArgument('joint_states_rate', default_value='30',
                              description='Hz; polled joint_states for the RViz robot model'),
        DeclareLaunchArgument('launch_robot', default_value='true'),
        DeclareLaunchArgument('launch_camera', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument('add_gripper', default_value='true',
                              description='xArm parallel gripper in the URDF'),
        DeclareLaunchArgument('serial_no', default_value="''",
                              description="RealSense serial, '' = first device"),
        DeclareLaunchArgument('color_profile', default_value='1280,720,30'),
        DeclareLaunchArgument('params_file',
                              default_value=PathJoinSubstitution([share, 'config',
                                                                  'calibration.yaml'])),
    ]

    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('xarm_api'), 'launch', 'xarm5_driver.launch.py'])),
        launch_arguments={'robot_ip': lc('robot_ip'),
                          'report_type': lc('report_type'),
                          'joint_states_rate': lc('joint_states_rate')}.items(),
        condition=IfCondition(lc('launch_robot')))

    # URDF: robot_state_publisher + joint_state_publisher fed by /xarm/joint_states
    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('xarm_description'), 'launch', '_robot_joint_state.launch.py'])),
        launch_arguments={'dof': '5', 'robot_type': 'xarm', 'hw_ns': 'xarm',
                          'add_gripper': lc('add_gripper')}.items(),
        condition=IfCondition(lc('launch_robot')))

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        launch_arguments={'camera_namespace': 'camera', 'camera_name': 'camera',
                          'serial_no': lc('serial_no'),
                          'enable_color': 'true', 'enable_depth': 'false',
                          'enable_infra1': 'false', 'enable_infra2': 'false',
                          'rgb_camera.color_profile': lc('color_profile'),
                          # RELIABLE publisher; backend subscribes RELIABLE depth 1
                          'color_qos': 'SYSTEM_DEFAULT'}.items(),
        condition=IfCondition(lc('launch_camera')))

    backend = Node(package='camera_extrinsic_calibration_core', executable='calibration_node',
                   name='camera_extrinsic_calibration', output='screen',
                   parameters=[lc('params_file')])

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
                arguments=['-d', PathJoinSubstitution([share, 'rviz', 'calibration.rviz'])],
                condition=IfCondition(lc('launch_rviz')))

    return LaunchDescription(args + [robot, description, camera, backend, rviz])
