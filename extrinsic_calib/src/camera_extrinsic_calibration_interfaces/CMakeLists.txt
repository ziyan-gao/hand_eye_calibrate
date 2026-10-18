cmake_minimum_required(VERSION 3.8)
project(camera_extrinsic_calibration_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CalibrationResult.msg"
  "msg/CalibrationStatus.msg"
  "msg/SampleSummary.msg"
  "srv/Configure.srv"
  "srv/Capture.srv"
  "srv/ListSamples.srv"
  "srv/DeleteSample.srv"
  "srv/SaveResult.srv"
  "srv/LoadSession.srv"
  "action/Solve.action"
  DEPENDENCIES builtin_interfaces geometry_msgs action_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
