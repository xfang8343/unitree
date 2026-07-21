# Unitree 强化学习学习记录

记录日期：2026-07-21

这个项目主要用于学习 Unitree G1 机器人在仿真中的强化学习训练、预训练策略回放、MuJoCo 部署配置，以及训练结果的观察方法。今天的核心目标是先把已有工程跑通，再理解每个命令背后对应的模型、配置、日志和评估指标。

## 项目结构

- `unitree_rl_gym-main/`：Unitree legged gym 主工程，包含训练脚本、环境配置、MuJoCo 部署脚本和机器人资源。
- `rsl_rl/`：PPO 强化学习算法库，训练时会被 `legged_gym` 调用。
- `unitree_rl_gym-main/deploy/deploy_mujoco/configs/g1.yaml`：G1 的 MuJoCo 部署配置。
- `unitree_rl_gym-main/logs/g1/`：训练后生成模型和日志的默认位置。

## 预训练模型回放

进入 MuJoCo 部署目录后运行：

```bash
cd /home/fx/unitree/unitree_rl_gym-main
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

G1 的 MuJoCo 配置文件在：

```text
/home/fx/unitree/unitree_rl_gym-main/deploy/deploy_mujoco/configs/g1.yaml
```

今天重点理解了这些字段：

- `policy_path`：加载哪个训练好的策略模型，例如 `deploy/pre_train/g1/motion.pt`。
- `xml_path`：加载哪个 MuJoCo 机器人场景模型，例如 `resources/robots/g1_description/scene.xml`。
- `kps` / `kds`：关节级 PD 控制器参数，决定关节跟踪动作时的力度和阻尼。
- `default_angles`：机器人默认站姿对应的关节角度。
- `num_obs: 47`：策略网络输入的观测维度。
- `num_actions: 12`：策略网络输出的动作维度，对应 12 个受控关节。
- `action_scale`：把策略输出缩放成目标关节角度的比例。
- `cmd_init`：初始速度命令，通常表示前后速度、左右速度、转向角速度。

## 训练 G1

进入训练工程目录：

```bash
cd /home/fx/unitree/unitree_rl_gym-main
```

开始一次较小规模的训练：

```bash
python legged_gym/scripts/train.py --task=g1 --headless --num_envs=128 --max_iterations=50
```

参数理解：

- `--task=g1`：选择在 `legged_gym/envs/__init__.py` 中注册的 G1 任务。
- `--headless`：不打开图形界面，适合只训练。
- `--num_envs=128`：并行仿真的环境数量，越大训练采样越快，但显存/内存压力也越大。
- `--max_iterations=50`：训练迭代次数，今天先用小迭代数验证流程。

训练完成后，模型通常生成在类似路径：

```text
logs/g1/日期_时间_/model_50.pt
```

## TensorBoard 看训练曲线

启动 TensorBoard：

```bash
tensorboard --logdir logs/g1 --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

今天重点看这些曲线：

- `Loss/value_function`：价值函数误差，过大或剧烈震荡说明 critic 学习不稳定。
- `Loss/surrogate`：PPO 策略更新相关损失，用来观察策略更新是否稳定。
- `Loss/learning_rate`：学习率变化。
- `Train/mean_reward`：平均奖励，整体越来越高通常说明策略在变好。
- `Train/mean_episode_length`：平均 episode 长度，越来越长通常说明机器人更不容易摔。
- `Episode/rew_tracking_lin_vel`：线速度跟踪奖励，越高说明越能跟上目标前进速度。
- `Episode/rew_tracking_ang_vel`：角速度跟踪奖励，越高说明越能跟上转向命令。
- `Episode/rew_orientation`：身体姿态相关奖励/惩罚，用于判断躯干是否更稳定。
- `Episode/rew_base_height`：机身高度相关指标，用于观察站姿和高度是否合理。
- `Episode/rew_action_rate`：动作变化率惩罚，惩罚变小说明动作更平滑。

## 回放训练结果

训练后可以运行：

```bash
cd /home/fx/unitree/unitree_rl_gym-main
python legged_gym/scripts/play.py --task=g1 --num_envs=16
```

`play.py` 会把训练好的策略加载出来，并且默认导出 JIT 策略文件，便于后续部署或在其他程序中调用。

## 今天遇到的问题与解决方案

### 1. 不知道预训练模型从哪里加载

一开始只是运行了 MuJoCo 部署命令，但不清楚模型路径在哪里配置。

解决方案：

查看 `deploy/deploy_mujoco/configs/g1.yaml`，确认 `policy_path` 指向预训练策略，`xml_path` 指向 G1 的 MuJoCo 场景文件。以后想换模型时，优先改 `policy_path`。

### 2. 不清楚 47 维观测和 12 维动作代表什么

配置里写了 `num_obs: 47` 和 `num_actions: 12`，但直接看数字不容易理解。

解决方案：

把它们和机器人控制问题对应起来理解：策略网络每一步读取机器人状态、速度命令、关节状态等观测，输出 12 个关节动作，再经过 `action_scale` 和 PD 控制器转成实际关节目标。

### 3. 训练时担心参数太大导致资源不够

强化学习训练会同时开很多仿真环境，`num_envs` 太大可能导致显存或内存不足。

解决方案：

先用小规模参数验证流程：

```bash
python legged_gym/scripts/train.py --task=g1 --headless --num_envs=128 --max_iterations=50
```

流程跑通后，再逐步增加 `num_envs` 和 `max_iterations`。

### 4. 不知道训练是否真的变好

只看终端输出不直观，很难判断机器人策略有没有进步。

解决方案：

用 TensorBoard 看 `Train/mean_reward`、`Train/mean_episode_length` 和各项 episode reward。判断策略时不要只看单条曲线，而要结合奖励、稳定性、速度跟踪和动作平滑程度一起看。

### 5. 训练结果和部署配置之间关系不清楚

训练生成的是 `logs/g1/.../model_*.pt`，MuJoCo 部署读取的是 yaml 里的 `policy_path`。

解决方案：

训练完成后，如果要在 MuJoCo 中回放自己训练的模型，需要把 `g1.yaml` 里的 `policy_path` 改成新生成的模型路径，或者把模型复制到一个固定的部署路径。

## 学习思路

今天形成的理解路线：

1. 先跑预训练模型，确认仿真、机器人模型和策略加载流程正常。
2. 再看 `g1.yaml`，理解策略模型、机器人模型、PD 控制器、观测维度和动作维度。
3. 接着跑小规模训练，先验证训练链路，而不是一开始追求长时间训练。
4. 用 TensorBoard 判断训练质量，重点观察奖励是否提升、episode 是否变长、动作是否更平滑。
5. 最后用 `play.py` 回放模型，把“训练得到的策略”与“仿真里的机器人行为”联系起来。

## 后续计划

- 继续阅读 `legged_gym/envs/g1/g1_config.py`，理解奖励函数、命令范围、地形设置和 PPO 参数。
- 对比预训练模型和自己训练的 `model_*.pt` 在 MuJoCo 中的表现。
- 尝试调整 `cmd_init`，观察机器人前进、横移、转向命令对步态的影响。
- 记录训练曲线截图和不同参数组合下的表现，形成更系统的实验笔记。
