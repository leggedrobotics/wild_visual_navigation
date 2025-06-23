# 《field_local_planner.launch文件使用指南》

## 简介

`field_local_planner.launch` 是一个ROS启动文件，用于为ANYmal机器人在野外环境中进行局部路径规划。它集成了一系列节点和配置，核心采用黎曼运动策略（Riemannian Motion Policies, RMP）算法，结合经过滤波处理的栅格地图（Grid Map）信息，实现机器人的平稳导航和障碍物躲避。

该系统可以根据纯几何信息或结合视觉信息进行地形可通性分析，并生成控制机器人移动的速度指令。

## 启动方法

可以通过标准的ROS `roslaunch` 命令启动该局部规划器。假设该功能包已正确配置在ROS工作空间中：

```bash
roslaunch wild_visual_navigation_anymal field_local_planner.launch [arguments]
```

### 可配置的启动参数 (Arguments)

启动时可以通过参数进行配置：

*   `debug` (bool, default: `false`): 是否启用调试模式。
*   `real_carrot` (bool, default: `true`): 关于目标点跟随方式的参数（具体影响需查阅 `field_local_planner_ros` 包）。
*   `base_inverted` (bool, default: `true`): 机器人基座是否反转180度。会覆盖 `rmp.yaml` 中的同名参数。
*   `traversability` (string, default: `visual`): 可通性分析模式。可选值为 `visual` 或 `geometric`。
    *   `visual`: 依赖于外部提供的视觉可通性信息层 (`visual_traversability`)，并结合几何特征。
    *   `geometric`: 纯粹基于高程图的几何特征（如坡度、高度差）进行可通性分析。
*   `output_twist_topic` (string, default: `/local_guidance_path_follower/twist`): 发布控制指令的话题名称。
*   `elevation_map_topic` (string, default: `/elevation_mapping/semantic_map_raw`): 输入的原始高程图话题。
*   `elevation_map_filtered_topic` (string, default: `/elevation_mapping/elevation_map_filtered`): 滤波处理后的高程图话题，供规划器使用。
*   `elevation_map_wifi_topic` (string, default: `/elevation_mapping/elevation_map_wifi`): 额外发布的简化版滤波高程图话题。
*   `elevation_map_filtered_filter_chain` (string, default: `$(find wild_visual_navigation_ros)/config/field_local_planner/$(arg traversability)/filter_chain.yaml`):
    主滤波链配置文件的路径。实际文件通常位于 `wild_visual_navigation_anymal` 包内对应 `traversability` 参数的子目录中。
*   `elevation_map_wifi_filter_chain` (string, default: `$(find wild_visual_navigation_ros)/config/field_local_planner/$(arg traversability)/filter_chain_wifi.yaml`):
    WiFi滤波链（简化输出）配置文件的路径。

## 输入机制

`field_local_planner.launch` 系统依赖以下输入：

### 1. 订阅的ROS话题

*   **机器人位姿 (Robot Pose):**
    *   话题: `/state_estimator/pose_in_odom`
    *   消息类型: `nav_msgs/Odometry` 或 `geometry_msgs/PoseStamped` (或带协方差的类似消息)
    *   用途: 提供机器人在 `odom` 坐标系下的当前位置和姿态。

*   **机器人速度 (Robot Twist):**
    *   话题: `/state_estimator/twist`
    *   消息类型: `geometry_msgs/TwistStamped` (或带协方差的类似消息)
    *   用途: 提供机器人的当前线速度和角速度。

*   **原始高程图 (Raw Elevation Map):**
    *   话题: 由 `elevation_map_topic` 参数指定 (默认为 `/elevation_mapping/semantic_map_raw`)。
    *   消息类型: `grid_map_msgs/GridMap`
    *   用途: 输入的原始环境地图数据。至少包含 `elevation` (高程)层。若 `traversability:=visual`，则此地图还需包含由视觉感知系统提供的 `visual_traversability` 层。

*   **目标点 (Goal Pose):**
    *   话题: `/initialpose`
    *   消息类型: `geometry_msgs/PoseStamped`
    *   用途: 指定局部规划器的导航目标点。常用于RViz中设置目标。

*   **操纵杆/遥控指令 (Joystick/Teleoperation Commands - 可选):**
    *   话题: `/cmd_vel`
    *   消息类型: `geometry_msgs/Twist`
    *   用途: 可用于手动控制或覆盖规划器的自主导航指令。

*   **当前吸引子 (Internal Attractor for Geodesic Field):**
    *   话题: `/field_local_planner/current_goal`
    *   消息类型: `geometry_msgs/PointStamped` 或 `geometry_msgs/PoseStamped`
    *   用途: 由规划器内部发布，供栅格地图滤波链中的测地线场（Geodesic Field）计算使用，作为生成测地距离图的目标点。

### 2. 配置文件 (Parameters)

*   **RMP核心参数:**
    *   文件: `wild_visual_navigation_anymal/config/field_local_planner/rmp.yaml`
    *   加载命名空间: `/field_local_planner`
    *   用途: 定义RMP算法的核心行为，包括机器人尺寸、速度限制、控制点、以及各种运动策略的权重、增益和度量函数。

*   **栅格地图滤波链参数:**
    *   文件 (根据 `traversability` 参数选择):
        *   `wild_visual_navigation_anymal/config/field_local_planner/visual/filter_chain.yaml`
        *   `wild_visual_navigation_anymal/config/field_local_planner/geometric/filter_chain.yaml`
    *   用途: 定义应用于原始高程图的处理步骤序列，以生成对RMP算法至关重要的信息层，如符号距离场 (SDF)、成本图和测地距离场。

## 核心原理

局部路径规划的核心原理是**黎曼运动策略 (RMP)**，该策略作用于一个经过精细处理的栅格地图之上。

### 1. 环境表示 (Processed Grid Map)

*   **输入处理:** 系统接收原始高程图。根据选择的 `traversability` 模式（`geometric` 或 `visual`），地图数据会经过一系列滤波和转换步骤（定义在对应的 `filter_chain.yaml` 中）。
*   **关键生成层:**
    *   **高程处理:** 原始高程数据被修复（填充空洞），并计算表面法线和坡度。
    *   **可通性估计 (`traversability`):**
        *   `geometric` 模式: 基于坡度、相对于机器人基座的高度等几何特征计算。
        *   `visual` 模式: 主要依赖外部提供的 `visual_traversability` 层，并进行修复和增强。
    *   **符号距离场 (SDF - `sdf`层):** 根据可通性信息计算，表示每个栅格单元到最近障碍物的距离。这是RMP障碍物躲避策略的关键输入。
    *   **成本图 (`cost`层):** 基于可通性信息生成，不可通行的区域具有高成本。
    *   **测地距离场 (`geodesic`层):** 在成本图上计算，表示从每个单元格到当前目标的（考虑障碍物的）最短路径距离。为RMP的目标跟踪策略提供梯度。
*   **输出地图:** 处理后的栅格地图（包含上述SDF、geodesic等层）被发布到 `elevation_map_filtered_topic` 话题，供RMP规划器使用。

### 2. 局部路径规划算法 (Riemannian Motion Policies - RMP)

*   **规划器插件:** 由 `field_local_planner::RmpPlugin` 实现。
*   **核心思想:** RMP通过定义多个“运动策略”来工作。每个策略根据机器人当前状态（位姿、速度）和处理后的栅格地图信息，计算出一个期望的加速度（或类似力的矢量）。
*   **控制点:** 这些加速度作用于机器人身体上定义的多个“控制点”（例如：中心、前左、前右、后左、后右，在 `rmp.yaml` 中配置）。这使得机器人能够做出更精细的反应。
*   **主要策略 (在 `rmp.yaml` 中配置):**
    *   **目标吸引:**
        *   `geodesic_goal`, `geodesic_heading`: 引导机器人沿测地距离场梯度（即地图上的最优路径）向目标移动。
        *   `goal_position`, `goal_orientation`: 直接将机器人吸引到最终目标的位置和姿态。
    *   **障碍物躲避:**
        *   `sdf_obstacle`: 基于SDF层产生排斥力，使控制点远离障碍物。
        *   `sdf_obstacle_damping`: 与障碍物交互相关的阻尼，防止过度反应。
    *   **运动阻尼与正则化:**
        *   `damping`: 产生加速度以抑制机器人的整体速度，帮助稳定运动。
        *   `regularization`: 可能用于惩罚过大的加速度，促进平滑控制。
*   **策略组合:** 所有活动策略产生的加速度通过加权和的方式组合（权重和增益在 `rmp.yaml` 中为每个策略指定）。每个策略的贡献也由特定的度量函数（如logistic、invlogistic）调整。
*   **速度生成:** 最终组合的加速度经过积分（积分时间 `integration_time` 在 `rmp.yaml` 中定义）转换为期望的线速度和角速度指令。

### 3. 数据处理与决策流程

1.  系统接收机器人位姿、速度和原始高程图（可能包含视觉可通性）。
2.  接收目标位姿。
3.  栅格地图滤波链处理原始地图，生成SDF、geodesic等关键层。
4.  RMP插件获取机器人当前状态和处理后的栅格地图。
5.  每个RMP策略计算其期望的加速度分量。
6.  这些加速度被加权求和。
7.  合加速度被积分生成目标速度。
8.  速度指令被发布以控制机器人。
9.  此循环以指定的控制频率（例如10 Hz）重复。

## 输出控制

系统通过发布ROS话题来控制机器人移动。

### 1. 发布的ROS话题

*   **话题名称:** 由 `output_twist_topic` 参数配置 (默认为 `/local_guidance_path_follower/twist`)。
*   **用途:** 该话题广播计算出的速度指令，机器人的底层运动控制器应订阅此话题以驱动机器人。

### 2. 控制指令格式与含义

*   **消息类型:** `geometry_msgs/TwistStamped` (由 `output_twist_type` 参数固定为 `twist_stamped`)。
*   **`geometry_msgs/TwistStamped` 结构:**
    *   `std_msgs/Header header`:
        *   `time stamp`: 指令生成的时间戳。
        *   `string frame_id`: 速度指令的坐标系，通常是机器人的基座标系 (例如 `rmp.yaml` 中定义的 `base_frame`，如 "base")。
    *   `geometry_msgs/Twist twist`:
        *   `geometry_msgs/Vector3 linear`:
            *   `float64 x`: 机器人x轴方向（前进/后退）的期望线速度。
            *   `float64 y`: 机器人y轴方向（侧向）的期望线速度（对于非完整轮式机器人通常为零或被忽略，但本项目中 `differential_mode` 默认为 `false` 且有 `max_linear_velocity_y`）。
            *   `float64 z`: 机器人z轴方向（上升/下降）的期望线速度（对于地面机器人通常为零）。
        *   `geometry_msgs/Vector3 angular`:
            *   `float64 x`: 机器人x轴（翻滚）的期望角速度（通常为零）。
            *   `float64 y`: 机器人y轴（俯仰）的期望角速度（通常为零）。
            *   `float64 z`: 机器人z轴（偏航/转向）的期望角速度。

*   **指令含义:**
    *   这些速度指令是在机器人自身坐标系（`base_frame`）下定义的。
    *   指令由RMP算法计算得出，旨在引导机器人前往目标点同时避开障碍物。
    *   生成速度受 `rmp.yaml` 中定义的 `max_linear_velocity_x/y` 和 `max_angular_velocity_z` 限制。
    *   指令以 `rmp.yaml` 中定义的 `control_rate` (默认10 Hz) 频率发布。

## 主要配置文件

*   `wild_visual_navigation_anymal/launch/field_local_planner.launch`: 顶层启动文件。
*   `wild_visual_navigation_anymal/config/field_local_planner/rmp.yaml`: RMP算法的核心参数配置。
*   `wild_visual_navigation_anymal/config/field_local_planner/visual/filter_chain.yaml`: “visual”模式下的栅格地图处理流程。
*   `wild_visual_navigation_anymal/config/field_local_planner/geometric/filter_chain.yaml`: “geometric”模式下的栅格地图处理流程。

通过调整这些配置文件和启动参数，可以针对不同的机器人特性和操作环境优化局部规划器的性能。
