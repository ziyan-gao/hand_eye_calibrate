# Camera Extrinsic Calibration（xArm5 + RealSense D455f，ROS 2 Jazzy）

实现 [Camera_Extrinsic_Calibration_Spec_v0_1.md](../Camera_Extrinsic_Calibration_Spec_v0_1.md)，算法沿用 [extrinsics_estimate.py](../extrinsics_estimate.py)（findChessboardCorners → cornerSubPix → solvePnP → calibrateHandEye），并补齐 spec 要求的配对、校验与持久化。

## 目录

```
docker/      Dockerfile、compose.yaml、compose.gpu.yaml、entrypoint.sh、record_versions.sh
src/
  camera_extrinsic_calibration_interfaces/  msg / srv / Solve action
  camera_extrinsic_calibration_core/        Python 后端（detector / handeye / session / node）+ 单元测试
  camera_extrinsic_calibration_rviz/        RViz2 Panel（C++/Qt，pluginlib）
  camera_extrinsic_calibration_bringup/     launch、calibration.yaml、calibration.rviz
data/        宿主持久化目录 → 容器 /data/calibration
```

## 约定

| 项 | 约定 |
|---|---|
| EEF | xArm `RobotMsg.pose`（TCP），TCP offset 必须在控制器中设为平行夹爪 TCP；每次 Capture 校验 `RobotMsg.offset` 与 `expected_tcp_offset` 一致，否则拒收 |
| 机器人输入 | `/xarm/robot_states`（xarm5_driver 默认 hw_ns=xarm），mm/rad → m，R = Rz·Ry·Rx（与旧脚本一致） |
| 相机输入 | `/camera/camera/color/image_raw` + `camera_info`（内参取自 ROS 驱动） |
| 棋盘 | Panel 中设置内角点 cols×rows 与格长 mm（默认 9×6、26.5 mm）；原点为外侧暗格对应的 0 号角点，避免 180° 翻转 |
| 结果 | eye_in_hand：`T_eef_from_camera_optical`；eye_to_hand：`T_base_from_camera_optical`；m + 四元数 xyzw + 矩阵 |
| TF | Solve 成功后发布 `/tf_static`：`link_tcp`（eye_in_hand）或 `link_base`（eye_to_hand）→ `camera_link`，由解得的 optical 外参与驱动的 `camera_link→optical` 合成；`camera_link` 已有其他父节点时拒绝发布。`publish_tf: false` 可关闭 |
| 默认方法 | PARK（报告中附全部方法对比；TSAI 在相对旋转接近 180° 时会失准） |

## 使用前必须确认

1. **夹爪 TCP offset**：在 UFACTORY Studio 设置平行夹爪 TCP，并把同一数值（mm, rad）写入 [calibration.yaml](src/camera_extrinsic_calibration_bringup/config/calibration.yaml) 的 `expected_tcp_offset`。当前默认 `[0,0,172,0,0,0]` 是 xArm Gripper 的出厂值，**需要你核对**。
2. `robot_ip` 默认 `192.168.1.234`（实测控制器地址；旧脚本中的 .232 已不可达）。

## 构建与运行（在宿主执行）

```bash
./run.sh                      # 日常：启动容器 → 增量 colcon build（几秒）→ launch
./run.sh robot_ip:=192.168.1.234 joint_states_rate:=30   # 参数直接传给 launch
./run.sh --rebuild            # 仅在修改 docker/Dockerfile 后需要
```

镜像只包含依赖。`src/` 与 `.colcon/{build,install,log}` 从宿主挂载，改代码后重新执行 `./run.sh` 即可，无需重建镜像。构建日志见 `.colcon/log/last_build.txt`。

机器人数据：`report_type` 保持 `normal`（5 Hz，包含 err 与 TCP offset；`dev` 不含这两项，会破坏 TCP 校验）。RViz 模型的流畅度由 `joint_states_rate`（默认 30 Hz）决定。

记录依赖版本（spec 3.1）：容器内 `record_versions.sh > /data/calibration/versions.lock`

## 操作流程

1. Panel 中设置 cols / rows / square size / mode → **Apply Config**（新建会话）。
2. 后端在机器人无错误后自动切到**手动模式**（mode 2，可拖动；`manual_mode_on_start` 可关闭），面板 **Manual Mode / Lock Pose** 按钮可切换。拖到新姿态后松手静止 → **Capture**。后端在点击后第一帧图像上检测棋盘，并在 ±0.25 s 窗口内检查机器人静止、时间差 ≤ 50 ms、TCP offset 正确，全部通过才计入样本。
3. 采集 15–25 个姿态，绕多个不平行的轴旋转；**Delete Last** 可撤销最新的一个样本。
4. **Solve** → 查看结果区的重投影 RMS、Y/Z 一致性误差和方法对比警告。
5. **Save Extrinsics** → `data/<session>/results/<name>/extrinsics.yaml` + `report.yaml`，不会静默覆盖已有结果。

会话数据（原图、逐样本 YAML）每次 Capture 后自动落盘，可用 **Load Session** 重新加载后再次求解。

## 测试

```bash
cd src/camera_extrinsic_calibration_core && python3 -m pytest -q test
```

覆盖：RPY/四元数约定、两种模式已知真值的反解（5 种方法）、方向错误可被检测、纯平移/同轴旋转/重复姿态拒收、渲染棋盘的原点一致性（含 180° 旋转）、会话增删与重载。

## 未做 / 未验证

- 未在本机执行 `docker build`，也未连接实机。
- 后端已在宿主 ROS Jazzy 上用模拟图像和模拟 RobotMsg 跑通端到端：Capture ×14、Delete Last、Solve、Save，求解误差 0.015 mm / 0.001°；运动中采集被拒收。
- 自动运动、验证集划分属于 spec 建议项，首版未实现。TF 只在本次运行 Solve 后发布，节点重启后需重新 Solve 或 Load Session 后再 Solve。
- 静止、时间差、重投影阈值均为工程初值，需要用实机数据重新标定。
