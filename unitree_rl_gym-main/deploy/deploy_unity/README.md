# G1 Unity Sim2Sim 策略服务

本目录用于把在 Isaac Gym 中训练并导出的 Unitree G1 12 自由度行走策略接入 Unity。

当前目标是完成：

```text
Unity 平地站立 -> 按速度行走 -> 与 Isaac Gym/MuJoCo 参数对齐
```

这里不包含真实机器人部署，也不依赖 `deploy/deploy_real/`。

## 1. 系统做了什么

整个闭环分为两部分：

```text
Unity                                            Python
------                                           ------
运行 G1 物理仿真
读取骨盆姿态、角速度和 12 个关节状态
生成期望速度 command
             -------- 原始状态 JSON -------->
                                                 构造 47 维观测
                                                 运行 TorchScript LSTM 策略
             <--- 12 维动作和目标关节角 -------
使用 ArticulationBody/PD Drive 执行目标关节角
推进下一次物理仿真
```

Python 负责策略和观测，Unity 负责机器人动力学。这样可以避免 Unity 和 Python 各自实现一套不同的观测缩放。

## 2. 目录内容

```text
deploy/deploy_unity/
├── configs/g1.yaml          G1 接口、策略路径、关节和控制参数
├── policy_server.py         TCP 策略推理服务
├── mock_unity_client.py     不启动 Unity 时使用的协议测试客户端
├── test_policy_server.py    观测排列和坐标旋转单元测试
└── README.md                本文档
```

机器人资产仍然使用：

```text
resources/robots/g1_description/g1_12dof.urdf
resources/robots/g1_description/meshes/
```

## 3. 为什么不能直接使用 model_8000.pt

`model_8000.pt` 是训练检查点，通常还包含训练器和优化器状态。Unity 联调使用的是 `play.py` 导出的 TorchScript 推理模型：

```text
logs/g1/exported/policies/policy_lstm_1.pt
```

从指定检查点导出：

```bash
cd /path/to/unitree/unitree_rl_gym-main
conda activate unitree-rl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python legged_gym/scripts/play.py \
  --task=g1 \
  --load_run=<run目录名> \
  --checkpoint=8000 \
  --num_envs=1
```

`play.py` 会把导出结果写到统一文件名 `policy_lstm_1.pt`。再次导出其他检查点会覆盖它。需要长期保存 8000 轮版本时，应另存为带版本号的文件，并通过服务端的 `--policy` 参数指定。

## 4. 环境要求

策略服务只需要：

```text
Python 3.8
PyTorch
NumPy
PyYAML
```

它不启动 Isaac Gym，因此正常运行服务时不需要 Isaac Gym 图形窗口。

确认策略文件可以加载：

```bash
python -c "import torch; p=torch.jit.load('logs/g1/exported/policies/policy_lstm_1.pt'); print(p)"
```

## 5. 启动 Python 服务

在项目根目录运行：

```bash
cd /path/to/unitree/unitree_rl_gym-main
conda activate unitree-rl

python deploy/deploy_unity/policy_server.py
```

成功后会看到：

```text
G1 Unity policy server
  listen: 0.0.0.0:9002
  policy: .../logs/g1/exported/policies/policy_lstm_1.pt
  device: cpu
  startup delay: 5.000 s
  protocol: one JSON object per line, 50 Hz
```

默认配置位于 `configs/g1.yaml`。常用覆盖参数：

```bash
python deploy/deploy_unity/policy_server.py \
  --host 0.0.0.0 \
  --port 9002 \
  --device cpu \
  --startup-delay 5 \
  --policy /绝对路径/policy_lstm_1.pt
```

CPU 足以运行这个小型 LSTM，且能减少 CUDA 环境差异。实测预热后的单次策略推理约为亚毫秒级，实际端到端延迟还取决于 Unity、网络和日志。

### 启动时等待 Unity 站稳

`configs/g1.yaml` 中的 `startup_delay_s: 5.0` 表示每次服务启动或收到
`reset=true` 后，先让 Unity 在默认站姿下保持 5 秒。等待期间 Unity 仍应以
50 Hz 发送状态；服务端返回：

```json
{"type":"hold","seq":1,"reason":"startup_delay","remaining_s":5.0}
```

`hold` 不是策略动作。当前 Unity 客户端应忽略它并保持默认站姿；到达 5 秒后，
服务端会清空 LSTM 和记忆中的上一动作，再从第一条正式 `action` 开始控制。
可通过 YAML 修改等待时长，或临时使用 `--startup-delay 0` 关闭。

### 查看 Unity 发送和 Python 返回的数据

使用下面的方式启动服务，会在终端打印每一条接收和返回的 JSON：

```bash
python deploy/deploy_unity/policy_server.py \
  --host 0.0.0.0 \
  --port 9002 \
  --print-messages
```

接收数据以 `RX` 开头，发送给 Unity 的结果以 `TX` 开头。50 Hz 每秒会产生约 100 行输出，长期打印可能影响实时性。联调时建议每 10 帧打印一次：

```bash
python deploy/deploy_unity/policy_server.py \
  --host 0.0.0.0 \
  --port 9002 \
  --print-messages \
  --print-every 10 \
  --startup-delay 3
```

正式性能测试时应关闭 `--print-messages`。

## 6. 不启动 Unity 的自测

先开一个终端启动服务，再开第二个终端：

```bash
cd /path/to/unitree/unitree_rl_gym-main
conda activate unitree-rl

python deploy/deploy_unity/mock_unity_client.py --steps 100 --realtime
```

模拟客户端会发送静止骨盆、虚拟关节状态和前进速度 `[0.3, 0, 0]`，然后显示推理时间和动作范数。

它只验证通信、模型加载和输出，不是真实物理仿真，不能用它判断机器人是否能站立。

修改测试速度：

```bash
python deploy/deploy_unity/mock_unity_client.py \
  --steps 100 \
  --command 0.0 0.0 0.5
```

运行观测单元测试：

```bash
cd deploy/deploy_unity
python -m unittest -v test_policy_server.py
```

## 7. 通信协议

协议使用 TCP，每一行是一个完整 JSON 对象，以换行符 `\n` 结束。一个请求必须对应一个响应。连接应长期复用，不能每 20 ms 重建一次。

第一版只允许一个 Unity 客户端，因为 LSTM 隐藏状态和上一动作属于同一台机器人。多个机器人需要为每个机器人维护独立策略状态。

### 7.1 获取接口配置

Unity 连接后建议先发送：

```json
{"type":"get_config"}
```

服务返回关节顺序、控制周期、默认关节角、PD 参数、单位和坐标约定。Unity 应检查返回的关节数量是否为 12。

### 7.2 推理请求

Unity 每 0.02 秒发送一次：

```json
{
  "seq": 1,
  "reset": false,
  "sim_time": 0.02,
  "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
  "base_angular_velocity": [0.0, 0.0, 0.0],
  "joint_position": [-0.1737, 0, 0, 0.3, -0.1263, 0, -0.1737, 0, 0, 0.3, -0.1263, 0],
  "joint_velocity": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "command": [0.3, 0.0, 0.0]
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `seq` | 请求序号，用于确认响应属于哪一帧 |
| `reset` | 场景刚重置时设为 `true`，只需发送一次 |
| `sim_time` | 当前 episode 从 0 开始的仿真时间，单位秒 |
| `base_quaternion_wxyz` | 骨盆相对协议世界坐标的姿态 |
| `base_angular_velocity` | 骨盆自身坐标系角速度，单位 rad/s |
| `joint_position` | 12 个关节角，单位 rad |
| `joint_velocity` | 12 个关节速度，单位 rad/s |
| `command` | `[前进速度, 左移速度, 左转角速度]` |

如果 Unity 更方便提供世界坐标系角速度，可以把 `base_angular_velocity` 换成：

```json
"base_angular_velocity_world": [0.0, 0.0, 0.0]
```

服务端会利用骨盆四元数转到机身坐标系。

`sim_time` 可以省略，此时服务端按每次成功推理增加 `0.02s`。建议 Unity 显式发送仿真时间，暂停或掉帧时步态相位更清晰。

普通推理请求可以省略 `type`，服务端会默认按 `infer` 处理。旧客户端使用的 `episode_time` 和 `base_angular_velocity_body` 仍然兼容。

### 7.3 推理响应

```json
{
  "type": "action",
  "seq": 1,
  "action": [12个数],
  "target_joint_position": [12个数],
  "command_used": [0.3, 0.0, 0.0],
  "inference_ms": 0.2
}
```

Unity 第一版直接使用 `target_joint_position`。它已经按下式计算完成：

```text
target_joint_position = default_joint_position + action * 0.25
```

请求中加入 `"debug": true`，响应会额外返回完整的 47 维 `observation`，适合联调，不建议长期打开。

若仍在 `startup_delay_s` 规定的站立等待时间内，响应类型是 `hold`，不包含
`target_joint_position`；Unity 应继续应用自己的默认关节目标，而不是将其当作错误。

### 7.4 重置和心跳

场景重置时可以单独发送：

```json
{"type":"reset","seq":100}
```

也可以在重置后的第一条 `infer` 中设置：

```json
"reset": true
```

这会同时清空：

- LSTM hidden state
- LSTM cell state
- 上一次动作
- 服务端内部 episode 时间

心跳请求：

```json
{"type":"ping","seq":101}
```

## 8. 47 维观测是怎样组成的

服务端严格按照 G1 训练环境构造：

| 索引 | 数量 | 内容 | 缩放 |
|---|---:|---|---|
| `0:3` | 3 | 机身坐标角速度 | `× 0.25` |
| `3:6` | 3 | 机身坐标投影重力 | 不缩放 |
| `6:9` | 3 | `[vx, vy, yaw_rate]` | `× [2, 2, 0.25]` |
| `9:21` | 12 | 当前关节角减默认关节角 | `× 1.0` |
| `21:33` | 12 | 关节速度 | `× 0.05` |
| `33:45` | 12 | 上一次策略动作 | 不缩放 |
| `45:47` | 2 | 0.8 秒步态相位的 `sin/cos` | 不缩放 |

总计：

```text
3 + 3 + 3 + 12 + 12 + 12 + 2 = 47
```

服务端使用 `float32` 输入策略。命令范围按训练配置限制到：

```text
vx       [-1.0, 1.0] m/s
vy       [-1.0, 1.0] m/s
yaw_rate [-1.0, 1.0] rad/s
```

初期联调建议只使用 `0.2～0.3 m/s` 和 `0.2～0.4 rad/s`。

## 9. 关节顺序

以下顺序是网络接口的一部分，Unity 不能使用组件遍历顺序代替：

```text
0  left_hip_pitch_joint
1  left_hip_roll_joint
2  left_hip_yaw_joint
3  left_knee_joint
4  left_ankle_pitch_joint
5  left_ankle_roll_joint
6  right_hip_pitch_joint
7  right_hip_roll_joint
8  right_hip_yaw_joint
9  right_knee_joint
10 right_ankle_pitch_joint
11 right_ankle_roll_joint
```

Unity 导入 URDF 后应按名字建立映射，再逐个做正方向测试。

## 10. 坐标系和单位

网络协议统一使用：

```text
右手坐标系
x：机器人前方
y：机器人左方
z：上方
四元数：w, x, y, z
长度：米
角度：弧度
线速度：m/s
角速度：rad/s
```

Unity 常用世界坐标与这个协议不同，而且 URDF Importer 可能在根物体上增加旋转。不要仅凭公式猜转换结果，应做三个校准测试：

1. 机器人直立且无旋转时，调试响应里的 `observation[3:6]` 应接近 `[0, 0, -1]`。
2. Unity 中让机器人向协议左侧移动时，发送的 `vy` 应为正。
3. 给单个关节发送正目标角，确认方向与 URDF 关节轴一致。

如果使用 Unity `ArticulationDrive.target`，要核实旋转关节目标使用的角度单位。通信协议始终使用弧度；若 Unity API 使用度，应在赋值边界执行 `rad × 180 / π`。

## 11. Unity 端建议实现方式

Unity 同事需要实现一个策略桥接组件，职责如下：

```text
Awake/Start
  1. 按名字找到 12 个 ArticulationBody
  2. 建立 TCP 长连接
  3. 请求 get_config 并校验接口

FixedUpdate（每 0.02 秒）
  1. 读取骨盆姿态和角速度
  2. 读取 12 个关节角和速度
  3. 转换成协议坐标和弧度
  4. 发送 infer
  5. 检查响应 seq
  6. 将 target_joint_position 应用到 12 个关节 Drive

Reset
  1. 重置骨盆和关节状态
  2. 清除 Unity 端旧动作
  3. 向 Python 发送 reset
```

正式实现不要在 Unity 主线程做可能无限等待的网络读取。建议后台线程负责 TCP，`FixedUpdate` 只提交最新状态并读取最新完整动作。应设置超时；超过约 100 ms 未收到新动作时停止增加速度指令，并进入暂停/重置流程。

## 12. Unity 物理参数对齐

第一版不追求复杂场景，先在无限平地对齐：

| 项目 | 参考值/要求 |
|---|---|
| 策略周期 | `0.02s`，50 Hz |
| Isaac Gym 物理步长 | `0.005s` |
| MuJoCo 物理步长 | `0.002s` |
| Unity 物理步长 | 可从 `0.005s` 开始，每 4 步调用一次策略 |
| 重力 | `[0, 0, -9.81] m/s²`，转换到 Unity 坐标 |
| 动作缩放 | `0.25 rad` |
| 初始骨盆高度 | 约 `0.8m`，再按碰撞体微调 |
| 步态周期 | `0.8s` |

PD 参考值已由 `get_config` 返回。Unity Drive 与 Isaac Gym/MuJoCo 的力矩实现并不保证完全等价，不能只复制数字后就认定参数一致。应记录关节响应曲线，逐步调整 stiffness、damping、force limit、摩擦和物理子步。

还需要核对：

- URDF 质量和惯量是否被正确导入
- 足底和地面碰撞体形状
- 足底/地面静摩擦和动摩擦
- 关节限位、最大力矩和阻尼
- 自碰撞设置
- Solver iterations 和接触偏移
- 初始姿态是否穿透地面

## 13. 分阶段验收

### 阶段 A：模型和接口

- G1 网格完整，没有缺失 STL
- 12 个关节名称和顺序正确
- 单关节正负方向正确
- Python 模拟客户端连续运行无错误
- Unity 能稳定完成 `get_config` 和 `ping`

### 阶段 B：平地站立

- 命令固定为 `[0, 0, 0]`
- 重置时发送 `reset=true`
- 机器人至少站立 30 秒
- 没有 NaN、关节爆转、骨盆持续倾倒

### 阶段 C：速度行走

按顺序测试：

```text
[ 0.2,  0.0,  0.0]  慢速前进
[-0.2,  0.0,  0.0]  慢速后退
[ 0.0,  0.2,  0.0]  向左横移
[ 0.0, -0.2,  0.0]  向右横移
[ 0.0,  0.0,  0.3]  左转
[ 0.0,  0.0, -0.3]  右转
```

每项记录目标速度、实际速度、站立时间、跌倒次数、关节角、动作和网络延迟。

### 阶段 D：Sim2Sim 对齐

在 MuJoCo 和 Unity 中使用相同策略、初始姿态和速度指令，比较：

- 起步方向和步频
- 骨盆高度和倾斜
- 实际前进/横移/转向速度
- 左右脚接触时序
- 12 个关节目标和实际响应
- 摔倒时间和失稳方式

不要一开始加入障碍物。平地闭环没有对齐时，复杂场景只会让问题更难定位。

## 14. 常见问题

### 策略文件不存在

```text
TorchScript policy not found
```

先运行 `play.py` 导出，或使用 `--policy` 指向正确文件。

### 机器人一开始就剧烈抖动

优先检查关节顺序、弧度/角度转换、关节正方向、默认角和 PD 参数，不要先重新训练。

### 机器人朝错误方向走

检查 Unity 到协议的坐标转换，以及 `command` 的定义：正 `vx` 前进，正 `vy` 左移，正 `yaw_rate` 左转。

### 机器人站立但迈步节奏异常

检查 `sim_time` 是否从零连续增加、调用频率是否接近 50 Hz，以及重置时是否清空了 LSTM。

### Unity 卡顿

不要每帧连接 TCP，也不要在主线程无限等待响应。复用连接并使用后台通信线程。

### 多次回放后模型效果不一致

确认 `policy_lstm_1.pt` 是否被后一次 `play.py` 导出覆盖。建议给策略文件增加版本名并记录对应训练 checkpoint。

## 15. 当前边界和下一阶段

当前策略只接收机身状态、关节状态、步态相位和速度指令。它没有目标位置、摄像头、深度图或障碍物距离，因此不能独立避障。

平地 Sim2Sim 完成后，推荐保持当前策略作为低层行走控制器，由 Unity 的 NavMesh、A* 或局部规划器把目标位置转换为：

```text
[vx, vy, yaw_rate]
```

之后再考虑把地形高度、射线或深度感知加入强化学习观测，训练感知行走策略。
