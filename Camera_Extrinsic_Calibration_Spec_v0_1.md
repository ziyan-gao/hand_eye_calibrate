# ROS 2 相机外参标定开发规格

**RViz2 Panel 插件与单容器开发环境**

版本 v0.1  ·  讨论与接口对齐稿  ·  基线 Ubuntu 24.04 + ROS 2 Jazzy

本项目将 xarm_ros2、realsense-ros、标定后端和 RViz2 Panel 安装到同一个开发容器中。操作者手动改变机器人姿态，通过面板采集配对的图像与末端位姿，再由 OpenCV 求解并保存相机外参。现有 SDK 测试脚本将作为接口适配和结果回归的依据。

本稿定义拟开发系统的行为和验收条件；当前未构建容器、未连接硬件、未验证用户测试脚本。明确要求、建议项和待确认项分别标记，避免把设计建议当成已确定需求。

## 1 已确定范围

| **项目** | **确定内容** |
|---|---|
| 前端 | RViz2 Panel 插件，不使用独立 Qt 窗口 |
| 输入 | 棋盘内角点列数与行数，例如 9 × 6；方格边长 mm；eye-in / eye-out 模式 |
| 操作 | Capture、Solve、Save Extrinsics、Delete Last |
| 观测 | 实时 EEF pose 与 camera 相对于棋盘的 pose；检测失败显示 null |
| 数值显示 | 全部位姿数值显示四位小数，数字变化和 null 不引起控件尺寸变化 |
| 开发环境 | Ubuntu 24.04、ROS 2 Jazzy；机器人与相机 ROS 包安装在同一容器 |

## 2 建议采用的包结构

| **包名** | **职责** |
|---|---|
| camera_extrinsic_calibration_interfaces | 自定义消息、服务、Solve action；不依赖硬件驱动 |
| camera_extrinsic_calibration_core | OpenCV 检测与求解、采集配对、数据集和结果管理；建议 Python |
| camera_extrinsic_calibration_rviz | C++ / Qt，继承 rviz_common::Panel，使用 pluginlib 注册 |
| camera_extrinsic_calibration_bringup | 驱动 launch、参数 YAML、RViz 配置、容器启动入口 |

建议增加：角点图像预览、样本列表与选中删除、标定质量指标、会话保存与加载。自动运动、自动采样轨迹、在线控制及相机内参重新标定不纳入第一版。

## 3 容器与依赖

### 3.1 版本策略

xarm_ros2 使用 jazzy 分支，必须递归初始化 xarm_sdk/cxx 子模块。realsense-ros 使用其 ROS 2 代码线，并选择与 librealsense 匹配的发布版本或固定提交。开发完成后记录基础镜像 digest、仓库提交、子模块提交及实际安装包版本。不得在每次构建时无记录地跟随分支最新版本。[S1][S2]

| **依赖** | **约定** |
|---|---|
| ROS 与系统 | Ubuntu 24.04 + ROS 2 Jazzy；宿主平台和 CPU 架构待确认 |
| 机器人 | xarm_ros2 / jazzy；首个适配目标暂按 UF850，实际机型待确认 |
| 相机 | realsense-ros、realsense2_camera、相容的 librealsense；型号待确认 |
| 前端 | RViz2、rviz_common、pluginlib、与 Jazzy RViz 匹配的 Qt 开发包 |
| 后端 | rclpy、tf2_ros、cv_bridge、sensor_msgs、geometry_msgs、OpenCV、NumPy、YAML |
| 构建 | colcon、rosdep、CMake、编译工具；按需构建依赖闭包 |

OpenCV 默认使用容器内与 ROS/cv_bridge 相容的系统包；不混装不同来源的同名 OpenCV Python 包。librealsense 只选择一种安装来源，版本配对在真实相机 smoke test 后锁定。MoveIt 和 Gazebo 可作为可选开发配置，不作为标定后端运行的必要组件。

### 3.2 宿主与设备连接

推荐首个验收平台为原生 Linux 宿主。单容器内运行两个驱动、后端和 RViz2；多个 ROS 节点仍为独立进程，不要求放入同一个 ROS component container。

USB：提供 RealSense 所需 USB/video 设备访问和权限，并验证重新插拔。udev 规则及内核驱动由宿主配置；容器共享宿主内核，不能通过镜像中的 DKMS 安装替代宿主内核准备。实际设备映射随宿主和 librealsense 后端确定。

网络：robot_ip 为外部配置；容器必须可达机器人控制器。Linux 首版建议 host networking；ROS_DOMAIN_ID 和 DDS 配置集中设置。

显示：提供经过授权的 X11 或 Wayland 显示连接；按显卡配置 GPU 设备或软件渲染。RViz 无法加载插件与 OpenGL 无法初始化应分别诊断。

### 3.3 工作区与产物

```
/opt/driver_ws       # 固定版本驱动 underlay
/workspaces/calib_ws # 可编辑标定 overlay
/data/calibration    # 宿主挂载的会话与结果目录
```

后续实现交付 Dockerfile、compose.yaml、依赖锁定清单、入口脚本和统一 launch。安装/构建无需连接硬件；运行配置通过挂载文件提供。源码和 /data 必须持久化，删除容器后样本仍可恢复。

## 4 坐标系与输入接口

### 4.1 统一变换方向

定义 T_A_from_B 为将 B 坐标系中的齐次点转换到 A 坐标系的 4 × 4 刚体变换。base 是机器人固定参考系；eef 是本次会话选择的法兰或 TCP；camera 是当前图像对应的 optical frame；board 是保持一致角点顺序的棋盘坐标系。所有求解与存储使用 m、rad，UI 使用 mm、deg。

| **对象** | **内部数据与用途** |
|---|---|
| 机器人观测 B_i | T_base_from_eef；不得混用 TCP、法兰、world offset 或不同参考系 |
| PnP 观测 C_i | T_camera_from_board；棋盘点转换到 camera optical frame |
| 用户要求的实时相机 pose | T_board_from_camera = inverse(C_i)，面板明确标注相对于棋盘 |
| Eye in hand 结果 X | T_eef_from_camera；相机固定于 eef，棋盘在 base 中固定 |
| Eye out of hand 结果 X | T_base_from_camera；相机固定，棋盘刚性固定于 eef |

### 4.2 机器人适配

默认使用配置指定的 ROS 输入，标准化为 geometry_msgs/TransformStamped：header.frame_id=base，child_frame_id=eef。来源选择为 xarm robot_states、时间戳 TF 或用户 SDK bridge；每个会话只采用一种权威来源。

已核对 Jazzy 分支的 RobotMsg：pose 是 TCP 位姿，位置 mm、姿态 rad，包含 TCP offset。适配层将位置乘以 0.001，并通过已验证的 SDK RPY 约定构造旋转。若选 flange，必须使用固定且已知的 TCP offset 转换；不能只把 TCP 改名为 flange。[S3]

UF850 默认驱动命名空间为 ufactory，因此 /ufactory/robot_states 可作为初始配置，不硬编码。SDK、ROS 输入和 FK/TF 的基准、工具偏置及角度约定须通过用户脚本与实机比对后确定。[S2][S4]

### 4.3 相机与棋盘

默认订阅 /camera/camera/color/image_raw 与 /camera/camera/color/camera_info；topic、序列号、分辨率和 frame 可配置。实时预览和 Capture 都使用 ROS 驱动提供的图像。SDK 相机模式为互斥替代模式，不能与 ROS 驱动同时占用同一设备。[S1]

只处理与当前图像尺寸、frame 和流类型匹配的内参。首版支持原始图像与受支持的 pinhole 畸变模型；其他模型明确拒绝或走专用适配。若支持 rectified 图像，须使用对应投影内参和零畸变，并禁止二次去畸变。

columns/rows 表示内角点数量；square_size_mm 为相邻角点间距。棋盘原点取编号 0 的内角点，x 沿列递增，y 沿行递增，z 由右手规则确定。保留原点/角点编号叠加；处理棋盘对称导致的角点编号翻转，不能在不同样本中任意换原点。

## 5 RViz2 Panel 行为

### 5.1 布局与显示

| **区域** | **组件与行为** |
|---|---|
| 配置区 | 内角点 columns/rows、square size、互斥模式选择、会话路径；高级设置折叠 |
| 图像区 | 当前图像、角点、编号 0 和棋盘轴；显示 detected / not detected / stale |
| 实时位姿区 | EEF in base 与 Camera in board，各固定六个位置和 RPY 数值栏 |
| 样本区 | 有效样本数量；建议提供列表、缩略图、时间差、重投影误差和选中删除 |
| 操作区 | Capture、Delete Last、Solve、Save Extrinsics；建议另有 Save/Load Session |
| 结果区 | parent/child frame、位置、旋转、样本版本、误差与有效/过期状态 |

面板建议英文按钮与字段名称，解释性提示语言可配置。第一版不提供机器人运动控制；操作者通过已有系统改变机器人姿态。图像和位姿未必同频刷新，UI 必须显示各自时间与有效性。

### 5.2 四位小数与固定空间

位姿数值使用等宽字体、固定小数点、显式符号及足够的固定字段宽度；例如位置 +00123.4567、角度 -00180.0000。统一 locale，使用小数点。字段宽度根据配置的范围和字体度量计算，正负号、数字更新、null 均不得改变布局。

检测失败时 Camera in board 六个字段显示 null；机器人掉线或数据超时时 EEF 字段也显示 null，并显示原因。禁止 NaN/Inf、禁止继续将旧值显示为当前有效值。极端值不能静默截断；超出字段范围显示固定宽度的无效标识。

RPY 显示采用固定轴 XYZ，即旋转矩阵 Rz(yaw) Ry(pitch) Rx(roll)，以 deg 表示；接近奇异姿态时提示。内部和保存使用旋转矩阵及归一化四元数，不用显示的舍入 RPY 作为求解输入。

### 5.3 按钮状态

Capture：后端就绪、图像/机器人新鲜、内参有效、完整棋盘检测成功且配对检查通过时可执行。Delete Last：存在有效样本时可用，移除最新采集的有效样本。Solve：通过样本有效性与运动检查后可用。Save：只允许保存当前数据集版本的有效求解结果。

Capture 和 Solve 执行中显示 busy/progress，并禁止冲突操作。GUI 不执行阻塞 SDK 调用、图像检测或求解。后台 ROS 回调通过 Qt queued signal 更新控件；插件卸载时取消请求并释放线程/订阅，不能阻塞或关闭整个 RViz。

### 5.4 RViz 配置

通过 Panels → Add New Panel 加载。实现 Panel 的 save/load，保存 topic、frame、模式和配置路径；样本保存在后端会话中，不嵌入 .rviz 文件。建议提供预置 calibration.rviz，恢复界面后重新向后端查询真实状态。[S6]

## 6 采集与数据集一致性

### 6.1 一次 Capture 的事务

点击 Capture 后，后端锁定配置版本，选取点击后到达的第一张可处理图像作为候选，记录其 header stamp、接收时间和帧标识。在该时间查找对应机器人位姿，完成角点检测、PnP 和检查后，原子提交完整样本。若配对失败，返回原因且不增加有效样本数。按钮不得仅保存一个旧缓存图像并同时读取另一个时刻的最新 EEF pose。

每个样本保存：原始无叠加图像、图像时间、机器人时间、配对时间差、两个接收时间、机器人原始数据及标准化变换、角点像素坐标、PnP 变换、内参/畸变、frame 名称、棋盘设置和质量指标。原始观测用于重新计算；UI 截图不作为求解数据。

### 6.2 时间与静止条件

首版采用静止采集。后端维护机器人位姿时间缓存；在 image stamp 对应的时刻读取 TF，或在统一时钟域内选择最近 robot_states 数据。静止窗口内位姿变化需低于配置阈值；只看机器人 state 不足以判定静止。

已检查的 xarm_driver 使用节点当前时间填写 robot_states.header.stamp，而非已确认的控制器采样时间；时间差只能反映软件时间对齐，不能宣称硬件同步。相机时间戳来源、传输延迟和时钟映射需实机核对。[S5]

建议起始参数：采集等待超时 2 s、静止窗口 0.5 s、允许时间差 50 ms、数据新鲜度 500 ms。均为工程初值，非设备保证或最终验收阈值。位置/角度静止阈值与重投影阈值待真实数据定标。不满足条件时停止采集并显示实际观测值。

若后续支持运动采集，需额外验证相机曝光时间、机器人反馈延迟、时钟偏移，并使用平移插值与四元数 SLERP；不能直接插值 RPY。离线回放可使用 use_sim_time，实机默认 false，不能混用 ROS time 与 wall time。

### 6.3 版本与删除

每个会话包含 session_id、config_revision 和 dataset_revision。添加、删除或启用/禁用样本均递增数据集版本。Solve 对固定快照求解；结果记录使用的配置和数据集版本。版本不匹配时结果标记 stale，必须重新求解后再保存当前外参。

已有样本时修改模式、棋盘规格、内参或 frame/source，会创建新会话或显式执行整组重处理；不得让历史样本隐式继承新设置。Delete Last 使用采集序号判断，不重用 sample_id；建议保留原图并将样本标记 excluded，便于撤销和追溯。

### 6.4 数据有效性

必需检查：角点数等于 columns × rows；数值有限；旋转为合法刚体旋转；棋盘点位于相机前方；尺寸/内参匹配；机器人状态可用；配对与静止条件通过。建议对模糊、高重投影误差、重复姿态和不一致角点编号给出警告或拒收原因。不能静默丢弃异常样本。

## 7 OpenCV 求解与结果验证

### 7.1 棋盘位姿

按固定角点顺序生成 z=0 的棋盘点，坐标单位 m。使用 OpenCV 检测并精化角点，通过 solvePnP 获得 C_i，即 T_camera_from_board。使用同一内参/畸变和 projectPoints 计算每张图的像素重投影 RMS。平面 PnP 的候选解应通过正深度、重投影和跨帧一致性检查；歧义未消除时拒绝该样本。[S7]

### 7.2 手眼求解约定

索引 i 表示一个有效样本；B_i 与 C_i 的定义见第 4 节。Eye in hand 的未知 X 为 T_eef_from_camera，静止棋盘的未知固定变换 Y 为 T_base_from_board。OpenCV 输入为每个 B_i 与 C_i 的旋转和平移。

```
Y_i = B_i @ X @ C_i
# Eye in hand: all Y_i should describe the same fixed board pose
```

Eye out of hand 的未知 X 为 T_base_from_camera，棋盘在 EEF 中的未知固定变换 Z 为 T_eef_from_board。calibrateHandEye 第一组输入必须使用 inverse(B_i)，第二组保持 C_i；输出解释为 T_base_from_camera，不使用 eye-in-hand 的命名直接保存。[S7]

```
Z_i = inverse(B_i) @ X @ C_i
# Eye out of hand: all Z_i should describe the same rigid board mount
```

建议默认使用一个经过用户数据验证的 OpenCV 方法；方法名可配置并存入结果。Tsai、Park 等方法比较属于建议项，不以方法间接近作为绝对精度证明。求解失败、退化、非有限输出或非法旋转均返回明确错误。

### 7.3 样本与运动要求

至少 3 个不同姿态且有绕非平行轴的相对旋转，这是 OpenCV 文档给出的必要采集条件；还需足够运动幅度与良好数值条件。仅检查样本数量不能保证可解。[S7] 建议初次采集 15–25 个具有多轴旋转和空间变化的有效姿态，具体数量由质量验证决定。

Solve 前检查重复姿态、相对旋转幅度与轴分布。配置最小旋转/运动阈值并报告检查值；严重退化时拒绝求解，覆盖不足时提示建议补采。禁止以只有平移或同轴旋转的样本通过简单数量门槛。

### 7.4 可解释的质量报告

分别报告图像重投影 RMS（px）和手眼一致性误差。对 Eye in hand 的 Y_i 或 Eye out of hand 的 Z_i，计算平移到其均值的距离，以及旋转到 SO(3) 平均旋转的测地角；显示每样本、RMS 与最大值，单位 mm/deg。指标实现必须记录参考均值和计算方法。

建议保留独立姿态作验证集：求解只使用训练样本，验证样本检查同一固定棋盘/安装变换的一致性。以上指标衡量内部一致性，不能当作外参真实绝对误差。若有独立测量或机器人对点测试，再单独报告实际任务误差。误差接受阈值待任务精度与用户测试数据确认。

## 8 后端接口与状态

### 8.1 设计原则

接口命名以下述 /calibration 为示例，允许统一 namespace。标定核心不直接依赖 xarm SDK；硬件专用内容位于适配器。GUI 与后端使用显式类型、请求 ID 和状态版本，不通过轮询临时文件通信。以下为拟议接口，最终 .msg/.srv/.action 待脚本对齐后冻结。

| **接口** | **建议类型与内容** |
|---|---|
| /calibration/eef_transform | geometry_msgs/TransformStamped；标准化 base ← eef，有效样本源时间戳 |
| /calibration/board_transform | geometry_msgs/TransformStamped；camera ← board，仅有效检测时发布 |
| /calibration/preview | sensor_msgs/Image；角点和坐标轴叠加的预览，保留源图像 stamp |
| /calibration/status | 自定义 CalibrationStatus；有效性、源时间、错误、版本、busy、样本数、结果状态 |
| configure | 服务；验证整组配置并返回 config_revision，不允许部分更新后继续工作 |
| capture | 服务；请求 ID 与预期版本；返回 sample_id、版本、质量或失败原因 |
| list_samples / delete_sample | 服务；完整样本摘要；按 sample_id 删除，last 由后端解析 |
| solve | action；输入快照版本与方法，反馈阶段，可取消；返回结果与质量报告 |
| save_result / save_session / load_session | 服务；后端路径、版本和覆盖策略；明确返回实际路径和成功/失败 |

标准 geometry_msgs 无法表示 null，故有效性由 status 显式携带。棋盘检测失效后不再发布旧 transform；GUI 使用状态和超时将字段置 null。preview 使用适合图像流的 QoS，CameraInfo 的订阅策略须与实际发布者相容，避免因 QoS 不匹配出现无数据。

### 8.2 状态与并发

后端主要状态：UNCONFIGURED、WAITING_FOR_DATA、READY、CAPTURING、SOLVING、ERROR。检测和源断开状态单独记录；短暂看不到棋盘不等于节点 ERROR。结果状态：NONE、VALID、STALE、FAILED。

所有修改数据集的操作串行化，Capture 请求不可重复提交。Solve 使用不可变快照；首版求解期间禁止修改样本。取消后不提交半成品，保留样本；求解失败不把上一次结果重新标为 VALID。RViz 重新加载面板不重置后端会话。

### 8.3 性能目标

建议位姿显示 10 Hz、图像预览 10–15 Hz，可配置限频；主线程始终可响应。慢速图像处理丢弃待处理旧帧，不能无限排队；采集请求使用其自身候选快照。具体吞吐量和求解时间在选定 CPU、分辨率与样本数量后测量，不能在 spec 中预先宣称已满足。

## 9 保存与 TF 发布

### 9.1 会话文件

建议每次采集自动持久化，再提供会话导入/导出。一个会话目录保存 session.yaml、samples/ 下的原图及逐样本 YAML、results/ 下的外参和质量报告。配置与样本版本固定关联；保存采用临时文件写入后原子替换，失败时不更新内存中的成功状态。

| **文件内容** | **最低字段** |
|---|---|
| session.yaml | schema_version、session_id、配置、frame、内参、模式、来源、数据集版本、样本索引、依赖版本 |
| sample metadata | sample_id、序号、excluded、原始/标准化 robot pose、C_i、角点、两个源时间与接收时间、dt、质量 |
| extrinsics.yaml | 模式、parent_frame、child_frame、transform_direction、单位、平移、四元数 xyzw、旋转矩阵/4 × 4 矩阵 |
| result provenance | method、OpenCV 版本、时间、配置/数据集版本、所用 sample_id、质量指标与检查阈值 |

存储 float64 完整有效精度，不将显示的四位小数写回数据。四元数 xyzw 的顺序明确；保存矩阵与四元数的一致性需检查。选择 output directory 和文件名；已有结果不得静默覆盖。Load Session 检查 schema、图像是否齐全及配置相容性，加载结果保持其原版本。

### 9.2 TF 的正确接入

Save Extrinsics 只写文件，不隐式修改机器人 URDF、RealSense 参数或当前 TF。发布 TF 为独立、默认关闭的后续选项。PnP 对应当前图像的 optical frame，不能把求解结果直接标成 camera_link。RealSense 自带 camera_link 到各传感器/optical frame 的内部关系。[S1]

若发布连接到 camera_link 的外部 TF，先通过驱动提供的固定内部变换，将解算的 optical 外参转换到 camera_link，然后以 eef 或 base 为 parent。camera optical frame 保留由 RealSense 驱动发布的 parent，不能额外给它添加第二个 parent。

```
T_parent_from_camera_link = (
    T_parent_from_optical @ T_optical_from_camera_link
)
# parent is eef (eye in hand) or base (eye out of hand)
```

发布前检查 frame 名称、已有 TF authority、重复边和闭环。机器人 URDF 若已有名义相机安装边，必须选择替换该来源或单独验证树，不能与新标定同时发布冲突边。预览状态中的棋盘位姿默认使用 topic，不要求每帧写入公共 TF 树。

### 9.3 用户测试代码接入

Robot SDK 脚本：保留读取姿态、单位和旋转转换逻辑的可验证部分，封装为 robot adapter。Camera SDK 脚本：先用于参考采集和离线回归；若启用 SDK live mode，则关闭 ROS 相机驱动并输出同一标准图像/内参接口。

OpenCV 脚本：提取检测、PnP、手眼求解及保存函数，隔离设备连接和交互逻辑。测试脚本：作为固定数据集回归基准，核对模式、变换方向和单位后比较矩阵。原脚本输出不能在未核对方向/偏置时直接视为真值。

## 10 验收与开发阶段

### 10.1 第一版验收条件

| **测试** | **通过条件** |
|---|---|
| 容器构建 | 无硬件完成镜像与所需包构建；记录版本；RViz 可发现并加载 Panel |
| 硬件输入 | 容器可接机器人和 RealSense；内参/帧匹配；源断开后规定超时内显示 null |
| 界面数值 | 正负、跨位数、零、null、异常值转换过程中数值控件几何尺寸不变 |
| 采集配对 | 成功样本原图与机器人位姿可追溯；无棋盘、超时、运动或不配对时不增加有效数 |
| 求解方向 | 两种模式用已知真值的合成数据检验 X；逆变换/单位错误能被测试发现 |
| 退化与异常 | 同轴旋转、纯平移、重复姿态、非法内参、PnP 失败得到明确失败/警告 |
| 数据生命周期 | Delete Last 正确作用于最新有效样本；变更使结果过期；Save/Load 可复现求解 |
| 真实数据 | 用户测试集与新包在统一约定下匹配；另采独立姿态验证，报告实际残差 |
| 持久化与运行 | 容器重启后数据完整；插件关闭不影响驱动；重复 TF 不被静默发布 |

数值容差分别定义：无噪声合成数据使用明确矩阵/旋转误差阈值；真实数据使用实验确定的 mm、deg、px 阈值。未获取设备与测试数据之前，不能把工程初值写成已通过的验收标准。

### 10.2 分阶段实施

阶段 A：核对用户脚本、设备型号、宿主系统、TCP/法兰与基准约定，冻结输入与数据 schema。阶段 B：构建容器并验证两个驱动，在 RViz 中加载空 Panel。阶段 C：接通实时图像/位姿、Capture、Delete Last 和会话持久化。阶段 D：接入两种模式求解、质量报告与 Save。阶段 E：离线回归、实机验收、锁定依赖并补充运行说明。

### 10.3 待用户提供或确认

仍需：机器人实际型号；RealSense 型号/序列号与选用图像流；宿主 Linux/Windows、CPU 架构和显卡；robot_ip；SDK 与 ROS 的 EEF 定义、TCP offset、world offset；棋盘规格与物理安装方式；已有采集、求解和测试脚本；输出目录和目标标定精度。上述信息不阻止规格初稿编写，但决定可执行容器与适配器实现。

已确认：RViz2 Panel、Jazzy、两个上游仓库与用户列出的六组核心功能。建议项尚待讨论：样本列表/选中删除、Save/Load Session、质量报告、方法比较和 TF 发布。

## 11 依据与版本记录

核对日期：2026年10月2日。下列资料用于确认上游接口；容器与实机兼容性尚未测试。仓库分支会变化，正式开发须锁定实际构建提交并重新核对接口。

[[S1] RealSense ROS 2 README 与输入话题及坐标系约定](https://github.com/realsenseai/realsense-ros/blob/9a11121700cb4780e273e34141f6402fe184321d/README.md)

核对代码线：ros2-master；提交 9a11121700cb4780e273e34141f6402fe184321d。README 支持 Ubuntu 24.04 / Jazzy，提供 color image、CameraInfo 和 optical frame 说明。此提交是检查记录，不是已验收的版本锁。

[[S2] xarm_ros2 Jazzy README 与 UF850 启动说明](https://github.com/xArm-Developer/xarm_ros2/blob/3dc2b5e8294758d96b54b15fa5920d581b7cbb3d/ReadMe.md)

核对代码线：jazzy；提交 3dc2b5e8294758d96b54b15fa5920d581b7cbb3d。README 说明 Jazzy/Ubuntu 24.04、递归子模块及 hw_ns 规则。Humble 分支内容不作为 Jazzy 已有功能的依据。

[[S3] RobotMsg 的 TCP 位姿与单位定义](https://github.com/xArm-Developer/xarm_ros2/blob/3dc2b5e8294758d96b54b15fa5920d581b7cbb3d/xarm_msgs/msg/RobotMsg.msg)

[[S4] UF850 launch 默认 ufactory 命名空间](https://github.com/xArm-Developer/xarm_ros2/blob/3dc2b5e8294758d96b54b15fa5920d581b7cbb3d/xarm_api/launch/uf850_driver.launch.py)

[[S5] xarm_driver 的 robot_states 位姿与时间戳发布](https://github.com/xArm-Developer/xarm_ros2/blob/3dc2b5e8294758d96b54b15fa5920d581b7cbb3d/xarm_api/src/xarm_driver.cpp)

[[S6] RViz2 Jazzy 的 rviz_common Panel 接口](https://github.com/ros2/rviz/blob/jazzy/rviz_common/include/rviz_common/panel.hpp)

[[S7] OpenCV calib3d 的 PnP 与 hand eye calibration 文档](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

OpenCV 4.13 文档用于算法输入方向说明，不表示容器必须安装 4.13。正式锁定的 OpenCV 版本需提供所选方法，并通过同一合成与真实数据回归。

### 版本变更

v0.1：整合六组核心功能、RViz2 Panel 决策、Jazzy 基线及两个驱动仓库；明确 TCP/光学坐标系、两种模式求解方向、时间配对、数据集版本和验收计划。
