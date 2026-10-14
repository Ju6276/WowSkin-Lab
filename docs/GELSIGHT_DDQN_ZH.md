# GelSightMujoco DDQN：Panda 抓放适配

入口为 `python -m wowskin_lab gelsight-ddqn`。直接加载 [GelSightMujoco](https://github.com/rlamprell/GelSightMujoco/tree/a060b0ec384b189e39d9576df753d381857bcc56) 固定版本的 `ddqn_model.py`、`neuralnetwork.py` 和 `replaybuffer.py`，在本项目 Panda＋GelSight 的 `pick_cube` 场景进行在线交互、训练和评估。当前接入原仓库默认的 **DDQN**；HDDQN 尚未接入。

这是上游 DDQN 学习器在本地任务上的适配。原仓库使用 UR5，Panda 的动作、相机和完整抓放评分有下文列明的差异。强化学习控制机器人动作，触觉图像仍由已有 GelSightMujoco 光学管线生成。训练从 ImageNet 图像特征和随机机器人策略开始，通过在线奖励学习；不使用之前录制的脚本示范、ACT 权重或物体位置作为策略输入。

## 环境准备

先安装主 README 的仿真／光学环境，再在仓库根目录执行：

```bash
.venv/bin/python scripts/setup_upstream.py --only gelsight_mujoco
.venv/bin/python -m pip install --target .tools/gelsight_ddqn_runtime \
  -r requirements/gelsight_ddqn.txt
.venv/bin/python -m wowskin_lab gelsight-ddqn prepare \
  --workspace outputs/gelsight_ddqn_workspace
```

TensorFlow 2.18.1、tf-keras 2.18.0、image-classifiers 1.0.0 及配套依赖放在独立目录。训练／评估进程优先使用该目录，并启用 Keras 2 兼容模式；主 `.venv` 的 protobuf／TensorBoard 不被降级。安装器可能对主环境的其他可见包打印依赖冲突；该隔离目录只用于此入口，不要把它永久添加到其他任务的 `PYTHONPATH`。每个准备、训练、评估输出目录都必须是新目录。

原始 checkout 保持干净。`prepare` 将三个 Python 源文件复制到工作区，生成独立的 `UPSTREAM_FIXES.patch`；`manifest.json` 保存原始／修改后 SHA256、补丁 SHA256 和修改说明。每次运行都核对摘要。

## 保留的算法与明确修改

保留原始 DDQN 的 epsilon-greedy、经验回放采样、在线网络选动作／目标网络估值的 Bellman 更新、Huber 损失及 Adam。图像分支是三个 ImageNet ResNet18，融合层为 1024／512／256，数值分支为 256／256／256。折扣率 0.99，epsilon 从 1 按每次更新乘 0.99996 衰减至 0.1，目标网络每 1000 条经验同步。

源文件补丁有三项：

- **Q 输出 softmax 改为 linear**：负距离奖励对应的回报可为负数，softmax 输出无法表达这些值。
- **多输入 replay 独立复制下一状态数组**：原代码对数组列表浅拷贝，写入下一状态时会覆盖当前状态。
- **Adam 使用 Keras 2 legacy 实现**：兼容原代码的 graph-mode `fit`。原优化器参数和损失保持不变。

本地运行器还在第一次更新前同步目标网络。真实终止（任务成功或掉落）不 bootstrap；人为时间上限仍 bootstrap。以上修正均记入运行报告，不能称为未经修改的原仓库实验结果。

## Panda 任务适配

输入采用上游可选的 `vision_tactile_raw` 组合：前方场景相机 RGB、左右 GelSight RGB、实测世界系 TCP xyz。图像按来源方式缩放到 32×32、保持 RGB 0–255，不额外做 ImageNet 输入归一化。来源的 `kinematics` 模式含物体真值；本入口未启用它。

动作是八个离散选择：世界系 X／Y／Z 正负移动各 5 mm，以及夹爪闭合比例加／减 0.1。姿态维持初始朝向。每个策略动作仅修改一次目标，随后按原 100 Hz 控制器保持；默认每 10 个控制 tick 决策一次，即 10 Hz。沿用现有 IK、速度／工作空间限制、夹爪控制与接触物理，没有物体吸附或位置复制。这个动作接口适配 Panda，区别于来源 UR5 的执行器增量。

奖励采用源环境 `_get_reward`：夹爪到物体、物体到目标两个欧氏距离之和取负。物体真值只参与奖励和评分。源奖励只表达距离，未显式奖励完整抓放的各阶段，因此保留以下更严格的本地成功标准：从源区抓起至少 6 cm，方块完全进入目标区，落在桌面，松爪并稳定保持 0.3 s。靠近、推到目标区或保持夹持均不能算成功。训练和评估报告分别保留距离、抓持、抬升和完整任务评分。

## 训练与独立评估

```bash
# 小规模流程试验：10 条至多 200 步的在线 episode
.venv/bin/python -m wowskin_lab gelsight-ddqn train \
  --workspace outputs/gelsight_ddqn_workspace \
  --output outputs/gelsight_ddqn_train \
  --steps 2000 --batch-size 16 --capacity 1024 --max-steps 200 --seed 0

.venv/bin/python -m wowskin_lab gelsight-ddqn eval \
  --workspace outputs/gelsight_ddqn_workspace \
  --checkpoint outputs/gelsight_ddqn_train \
  --output outputs/gelsight_ddqn_eval --episodes 5 --seed 1000

# 相同独立种子上的均匀随机动作参考
.venv/bin/python -m wowskin_lab gelsight-ddqn eval \
  --workspace outputs/gelsight_ddqn_workspace \
  --checkpoint outputs/gelsight_ddqn_train \
  --output outputs/gelsight_ddqn_random --episodes 5 --seed 1000 --random
```

示例命令缩小了训练步数、batch 和容量；CLI 默认 batch=128、容量=10000 与上游默认相同。这个小规模试验用于检查真实更新和闭环运行，不预先保证任务收敛。

训练保存 `run.json`、逐步 `steps.jsonl`、`policy.h5`、重载校验观测和适配代码快照。checkpoint 用于推理，不包含可续训的 replay／优化器状态。评估先核对源文件、环境代码和 checkpoint SHA256，再对照保存前后的 Q 输出；拒绝使用训练中出现过的种子。每条评估保存三路图像拼接 MP4 及 `evaluation.json` 中的实际成功率。

数据、模型和 MP4 均保留在忽略的 `outputs/`，不上传大文件。轻量检查结果可另行保存在仓库文档中。

## 2026-10-14 实测结果

按上述小规模配置完成 2,000 步真实在线交互、1,984 次网络更新，共 10 个训练 episode（种子 0–9），耗时约 697 秒。损失均为有限值，最后 epsilon 为 0.9237；训练期间没有实际抬起方块，完整抓放成功数为 0。

在独立种子 1000–1004 上，贪心 DDQN 和均匀随机动作分别完成 5 条、每条 200 步评估，完整抓放均为 **0/5**。DDQN 的平均累计奖励为 -133.187，随机参考为 -56.042（该负距离奖励越大越好）。DDQN 主要选择向上移动，这轮小规模训练没有产生有效抓放策略，奖励表现也劣于随机参考。不能据此宣称算法已收敛或可以执行任务。

checkpoint 保存前后的 Q 输出校验通过；10 个评估 MP4 全部完整解码，每个恰好 200 帧。91 项单元测试和 3 项上游来源测试通过，无跳过；本次未重跑完整物理测试组。逐种子指标、动作分布、来源版本和报告摘要保存在 [validation.json](../examples/gelsight_ddqn/validation.json)。本机训练及两组评估分别位于 `outputs/gelsight_ddqn_20261014/train_2000/`、`eval_2000/`、`random_2000/`。

当前完成的是可复现的算法接入及小规模流程验证。若继续追求成功率，需要单独开展更长训练和任务学习实验；本次未修改来源距离奖励来制造成功结果。
