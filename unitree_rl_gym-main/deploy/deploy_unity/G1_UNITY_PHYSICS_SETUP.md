# G1 Unity 基础物理与关节参数

本文件对应当前强化学习策略使用的 `g1_12dof.urdf`，而不是完整 29 自由度 G1。

机器可读参数清单：

```text
deploy/deploy_unity/configs/g1_unity_physics.json
```

该 JSON 的数据来源是：

```text
URDF：质量、惯量、连杆安装位置、关节轴、硬限位、最大力矩、最大速度
G1 强化学习配置：默认关节角、Kp、Kd、动作缩放、控制周期
```

JSON 中的刚体数据覆盖骨盆和 12 个受控关节直接带动的连杆。完整 URDF 还包含固定的躯干、足部和其他附件，它们的质量、惯量、视觉与碰撞体同样会影响站立。因此 Unity 应完整导入 `g1_12dof.urdf`；该 JSON 用于校验和配置策略接口，不应用它单独手工拼出整台机器人。

## 1. Unity 必须设置的参数

### 刚体和碰撞

优先用 Unity URDF Importer 导入 `g1_12dof.urdf`。导入后检查而不是随意改写：

```text
每个 Link 的 mass
每个 Link 的 center of mass
每个 Link 的 inertia tensor
URDF collision 模型
父子连杆关系和 joint origin
关节旋转轴
```

如果 Importer 没有正确导入，则从 JSON 的 `root_link` 和每个 joint 的：

```text
child_link_mass_kg
child_link_inertia_urdf_kg_m2
origin_xyz_m
origin_rpy_rad
axis_urdf
```

手动补齐。URDF 惯量包含 `ixy/ixz/iyz` 非对角项，不能只把三个对角值随意填入；应优先让 Importer 完成惯量张量转换。

### 12 个关节 Drive

Unity 端必须按 JSON `joints[index]` 的顺序，通过 joint `name` 找到对应 `ArticulationBody`。

每个关节至少设置：

```text
lower limit      = limit_deg[0]
upper limit      = limit_deg[1]
stiffness        = stiffness
damping          = damping
force limit      = effort_limit_nm
target           = 目标弧度 × Mathf.Rad2Deg
```

`effort_limit_nm` 是最大力矩，不能用它代替 `stiffness`。

用于策略的目标角始终为弧度：

```text
target_rad = target_joint_position[i]
target_deg = target_rad * Mathf.Rad2Deg
```

例如：

```text
左膝默认目标：0.3 rad = 17.1887 deg
左踝 pitch 默认目标：-0.1263 rad = -7.2365 deg
```

## 2. 可直接使用的 12 关节控制表

| 索引 | 关节 | 下限/上限 (deg) | 默认 (deg) | Kp | Kd | 最大力矩 (N*m) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | left_hip_pitch_joint | -145.0 / 165.0 | -9.9523 | 150 | 2 | 88 |
| 1 | left_hip_roll_joint | -30.0 / 170.0 | 0 | 150 | 2 | 139 |
| 2 | left_hip_yaw_joint | -158.0 / 158.0 | 0 | 150 | 2 | 88 |
| 3 | left_knee_joint | -5.0 / 165.0 | 17.1887 | 225 | 4 | 139 |
| 4 | left_ankle_pitch_joint | -50.0 / 30.0 | -7.2365 | 80 | 3 | 50 |
| 5 | left_ankle_roll_joint | -15.0 / 15.0 | 0 | 30 | 2 | 50 |
| 6 | right_hip_pitch_joint | -145.0 / 165.0 | -9.9523 | 150 | 2 | 88 |
| 7 | right_hip_roll_joint | -170.0 / 30.0 | 0 | 150 | 2 | 139 |
| 8 | right_hip_yaw_joint | -158.0 / 158.0 | 0 | 150 | 2 | 88 |
| 9 | right_knee_joint | -5.0 / 165.0 | 17.1887 | 225 | 4 | 139 |
| 10 | right_ankle_pitch_joint | -50.0 / 30.0 | -7.2365 | 80 | 3 | 50 |
| 11 | right_ankle_roll_joint | -15.0 / 15.0 | 0 | 30 | 2 | 50 |

## 3. Unity 初始化顺序

1. 导入 URDF 和 meshes，确认质量、碰撞体和关节层级存在。
2. 暂时关闭重力，逐个关节施加小的正/负目标角，确认关节方向。
3. 设置上表的关节硬限位、stiffness、damping、force limit。
4. 设置全部 12 个默认角。Unity Drive target 使用角度值。
5. 读取 Unity 实际关节角，并转换为 rad；应接近：

```text
[-0.1737, 0, 0, 0.3, -0.1263, 0,
 -0.1737, 0, 0, 0.3, -0.1263, 0]
```

6. 开启重力和地面碰撞。初始骨盆位置按协议为 `[0, 0, 0.8]`，但 Unity 使用 Y 轴向上，且 Importer 可能旋转根节点；应以脚底刚好接触地面、没有穿透为准。
7. 仅用默认姿态 PD 站立 10 秒，确认机器人不会立即倒下。
8. 再接入 Python 策略，第一帧使用 `reset=true`、`sim_time=0`、`command=[0,0,0]`。

## 4. 时间、地面与网络

```text
Unity physics step：0.005s
策略请求周期：0.02s
每4次物理步请求一次 Python
重力：Unity 世界坐标 [0, -9.81, 0]
初期地面反弹：0
初期足底/地面摩擦：0.8 到 1.0，再做 Sim2Sim 校准
```

当前策略训练时随机化摩擦范围为 `0.1 到 1.25`，因此 0.8 到 1.0 是调试起点，不是唯一正确值。

## 5. 推荐的 Unity Drive 伪代码

```csharp
void ApplyTarget(ArticulationBody joint, float targetRad,
                 float lowerDeg, float upperDeg,
                 float stiffness, float damping, float forceLimit)
{
    ArticulationDrive drive = joint.xDrive;
    drive.lowerLimit = lowerDeg;
    drive.upperLimit = upperDeg;
    drive.stiffness = stiffness;
    drive.damping = damping;
    drive.forceLimit = forceLimit;
    drive.target = Mathf.Clamp(targetRad * Mathf.Rad2Deg, lowerDeg, upperDeg);
    joint.xDrive = drive;
}
```

这段代码假设 URDF Importer 已把每个单自由度关节映射到正确的 Unity Drive 轴。若 Importer 使用的不是 `xDrive`，必须按导入后的实际轴调整，但协议中的关节顺序、弧度单位和限位数值不变。

## 6. 当前日志对应的检查

若 Python RX 中的膝关节长期接近 `0 rad`，而不是约 `0.3 rad`，优先检查：

```text
Python 返回 rad，Unity Drive target 是否误当成 deg
默认站姿是否确实设置
关节正方向是否相反
Unity 读取的 joint position 是否按 rad 发送
```

在默认姿态、关节读数和 50 Hz 闭环未确认前，不要用策略动作大小判断模型好坏。
