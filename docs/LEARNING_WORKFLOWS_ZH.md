# ViSk 与 FTF：从当前采集开始

采集入口见 [README](../README.md)。三个任务共享一套 LeRobot v2.1＋100 Hz raw 数值附件，不需要同时开两套录制器。当前 cube 示范保留在 `data/episodes/pick_cube/`；旧试验和策略权重已移出项目。以下训练命令需采集完成后手动运行。

## 当前采集内容

- 相机：固定 `1/2`＋腕部 `52`，三路 640×480、30 Hz RGB-D。深度无损保存，腕部外参逐帧保存；配置见 [cameras.json](../configs/cameras.json)。
- 机器人：基座系 TCP 位姿、实际下发目标、七关节位置／速度、TCP 线／角速度。
- 夹爪：连续实测闭合比例与连续目标分开保存，0 开、1 闭；另外保留物理驱动关节及单位。Panda、Robotiq 使用相同语义。
- 触觉：单块五芯片 XYZ，共 15 路磁场；raw 保存基线和质量字段。拟合偶极子没有完成真机标定。
- instruction、时间戳、相机标定、控制参数、任务自动评分和人工保存结果。

B 复位后开始录制；绿色 SUCCESS 不会自动停录，Enter 保存成功，N 保存失败。自动评分和人工标签分别保存。成功区域只是操作者辅助图形，不进入相机或策略输入。保存成功后图像只留在 LeRobot，raw 的 H5 是 100 Hz 数值附件，不是第三套数据集。控制或相机配置不同不能追加到同一数据集。

夹爪宽度的统一约定：后续宽度字段对齐真机驱动的反馈／命令语义。`gripper_width_m` 表示当前反馈开口，`gripper_target_width_m` 表示实际下发的目标开口；均为米，分别属于 state 和 action。Franka 真机对应 `GripperState.width`，仿真从关节反馈转换到同等零位、量程和单位；不把指腹形变后的表面最短距离作为采集要求或策略输入。[Franka 官方字段说明](https://frankarobotics.github.io/libfranka/0.15.0/structfranka_1_1GripperState.html)。Robotiq 按其驱动反馈与机构转换建立相同语义，不能把转轴角度比例直接当宽度。

这是已明确的采集约定，**当前代码仍保存闭合比例和原始驱动关节，尚未增加上述米制宽度字段**。没有更改现有 episode。需要比例的算法在导出时转换，并保留实测状态与目标指令的区别。

## ViSk：导出与训练

以下命令在新采集成功后运行，输出目录必须不存在：

```bash
.venv-reduced/bin/python -m wowskin_lab dataset-export data/episodes/pick_cube --format visk --output data/exports/visk_cube
.venv-reduced/bin/python -m wowskin_lab official prepare visk \
  --workspace data/exports/visk_cube/workspace --data-root data/exports/visk_cube --task pick_cube
.venv-reduced/bin/python -m wowskin_lab official process-visk-sim \
  --workspace data/exports/visk_cube/workspace
```

`--episodes capture_000000 capture_000002` 可选择演示。喂食任务换成 `feeding`，线材换成 `cable` 或 `cable_insert`。不同控制配置的数据须分别导出，不与默认 cube 数据混合。

ViSk 原转换器生成 `processed_data_pkl_aa/pick_cube.pkl`，其中含图像、位姿、夹爪状态、触觉及统计量。本地桥接器按真实时间戳对齐，并用录制起止事件截取片段。原加载器继续执行采样、基线扣除、归一化和动作块构造。腕部图像沿用官方转换器裁掉右侧 20% 再缩放的规则，在线输入也执行同一裁剪。

当前训练观测是 `pixels1/2/52`＋`sensor0`，输出为 10 步×7 维动作块。七维是基座系平移增量、末端局部旋转增量、夹爪绝对闭合指令。当前 `aux_keys=[sensor0]` 不编码本体状态；状态仍被保存，并用于构造运动标签。深度、语言和关节速度暂不作为原版 ViSk 的输入。

**记录在案的本地差异：**

- 论文是单侧 AnySkin，公开代码默认双组；`PAPER_SINGLE_SENSOR.patch` 选择一组，不补零或复制数据。
- 论文四相机，本项目按用户要求三相机；自动读取导出元数据并生成 `THREE_VIEW_CAMERAS.patch`。先前四路布局仍保存在 `configs/cameras.json`。
- 仿真反馈开度不能代替闭合指令。`MUJOCO_ACTION_LABELS.patch` 保留 measured gripper 作为状态，动作标签取录制的 `commanded_gripper`；缺少该字段会报错，不反推或强制全闭。公开数据对照可显式选择 `--visk-action-profile published-state`。
- Franka/MuJoCo 是本地环境接口，不是论文的原 XArm 硬件。模型、损失、归一化、动作块和官方采样逻辑没有替换。
- 部署频率尚未与论文对齐：论文 IV-A 写 10 Hz；当前 `visk_sim.py` 根据本地 30 Hz 图像和公开加载器默认 `subsample=5` 使用 6 Hz（仿真时间）。这属于本地执行时序选择，不是论文规定。改变它前需一并核对标签时间间隔，不能只改执行频率。
- 论文 III-A 在遥操作速度方向上加入小角度随机扰动；当前本地采集没有复现该示教扰动。`--ideal` 控制的是触觉读出噪声等，不是该方向扰动。任务与数据量也不是论文实验配置。[论文 III-A / IV-A](https://arxiv.org/html/2410.17246v2#S3.SS1)

补丁、来源和 SHA256 保存在 workspace 的 `manifest.json`。官方参考 checkout 不修改。这是带明确数据／环境适配的官方算法流程，不能称为硬件、任务和论文成绩的完整复现。

以下是以后训练用的命令，本次**不启动训练**：

```bash
.venv-reduced/bin/python scripts/train_visk_mac.py \
  --workspace data/exports/visk_cube/workspace \
  --output outputs/training/visk_cube_three_view --steps 10000 --save-every 3000
```

默认 MPS、batch 256、subsample 5、动作块 10。调用原 `WorkspaceIL.train()` / `BCAgent.update()`；Mac 运行差异写入 `status.json`：不支持的 MPS 运算走 PyTorch CPU fallback，float64 在传入 MPS 前转为原 update 要求的 float32，DataLoader 默认单进程。它们不保证与官方 CUDA 位级一致。按原循环零起始步号，`3000.pt` 在第 3001 次更新后保存；另存最后更新完成的 `10000.pt`。TensorBoard／CSV 与 checkpoint 频率独立。

## ViSk MuJoCo 实时闭环

训练完成后：

```bash
.venv-reduced/bin/mjpython scripts/eval_visk_sim.py \
  --training outputs/training/visk_cube_three_view \
  --checkpoint outputs/training/visk_cube_three_view/snapshot/10000.pt \
  --output outputs/evaluation/visk_cube_three_view_01 --viewer
```

执行原版 `WorkspaceIL.eval()`、`BCAgent.act()`、checkpoint 归一化和时间聚合。每次从实时三路图像与触觉产生一个 7 维动作，再分摊到 100 Hz 控制周期，持续累积运动目标。零增量保持目标，不在每次推理时丢弃尚未跟上的位移。反馈开度不覆盖预测的夹爪命令；底层限幅、实际执行量和评分写入结果。原训练片段不参与闭环回放。

读取采集元数据中的物理／控制配置；评估和采集共用任务成功判定。显示用独立 MuJoCo data，窗口 reset 不改写运行中的策略物理状态。目前支持 cube 和喂食闭环；线材策略执行入口尚未接入。旧错误夹爪标签的模型不能通过更换接口变成正确模型；新采集仍需要新训练和独立测试。控制器对照能完成动作不等于策略学会任务。

接口回归覆盖：官方转换器／加载器保留实测夹爪状态并使用真实指令作为标签、旋转基座下的坐标转换、增量跨控制周期只执行一次、跟踪滞后和零增量时保留累计目标，以及显示窗口 reset 不修改真实物理状态。`tests/unit/test_visk_action_transport.py` 与 `tests/official/test_visk_command_labels.py` 共 4 项测试已于 2026-10-11 通过；它们不验证新三相机策略成功率，也不证明本地部署频率等同论文。

## FTF：机器人记录到视觉点轨迹

FTF 使用**稀疏三维关键点轨迹＋触觉接触历史**，不是稠密点云。机器人参考点由官方位姿几何构造；物体点来自人工标注和视觉跟踪，点数取决于任务，不能固定成“17 个点”。15 路触觉完整保留，官方机器人转换器取首个芯片扣基线后的 XYZ 范数作为标量接触量，不是牛顿力。

FTF 从同一份记录中选择固定相机 1、2，腕部数据仍留在公共数据集。深度不自动替代其原版双目三角化。

```bash
.venv-reduced/bin/python -m wowskin_lab dataset-export data/episodes/pick_cube --format ftf --output data/exports/ftf_cube
.venv-reduced/bin/python -m wowskin_lab official process-ftf-robot \
  --data-root data/exports/ftf_cube --task pick_cube
.venv-reduced/bin/python -m wowskin_lab official convert-ftf-robot-initial \
  --data-root data/exports/ftf_cube --task pick_cube \
  --calib-path data/exports/ftf_cube/extracted_data/pick_cube/demonstration_0/sim_camera_calibration.npy
.venv-reduced/bin/python -m wowskin_lab official prepare-ftf-annotation \
  --data-root data/exports/ftf_cube --task pick_cube --output data/exports/ftf_cube/annotation
.venv-reduced/bin/python -m notebook data/exports/ftf_cube/annotation/label_points.ipynb
```

预处理使用固定版本 `process_data_robot.py`、`convert_to_pkl_robot.py`。初始 PKL 只供标注，`training_ready=false`。短到未离开初始位姿的记录可能被官方预处理丢弃，不修改它的时间戳或姿态让其通过。

接下来必须完成：

1. notebook 在两视角首帧按同一物理点、同一顺序标注并保存。
2. CUDA 上用官方 DIFT 匹配各段起点、CoTracker 跟踪、双目三角化恢复三维轨迹；运行转换器 `--process_points`。先移走初始 PKL，避免原脚本追加混入未跟踪数据。
3. 检查轨迹、点数配置及官方机器人参考点与本地 TCP 配准。
4. 机器人分支运行 `train.py agent=p3po suite=p3po dataloader=p3po`，之后对接官方触觉反馈控制。

Mac 可做导出、视频预处理、初始 PKL 与标注。固定版本 DIFT/CoTracker 路径直接依赖 CUDA/xformers，当前 Mac 尚未跑完视觉处理，也未验证 FTF MuJoCo 策略闭环。不会用仿真物体真值、八个角点或简化跟踪器代替官方视觉流程。自建任务需准备点数一致的 YAML，用 `official prepare ftf-robot --task-config PATH.yaml` 生成可审查工作目录，命令见其 `COMMANDS.md`。

## FTF 人手示范分支

人佩戴触觉手套直接操作物体，双相机＋同步触觉记录；不需要用机器人遥操作来替代人的动作。公开 Franka-Teach 同时提供 human/robot 模式：

```bash
# 在官方 Franka-Teach 目录，且已接好它依赖的机器人／相机／触觉服务
python teleop.py teleop_mode=human
python collect_data.py collect_img=true collect_state=true collect_reskin=true demo_num=0
```

输出 AVI／时间戳、触觉 H5、`states.pkl` 和 `commanded_states.pkl`。公开 human 入口仍调用机器人 reset／状态服务，当前没有在本机接好独立人手采集，不能称为已完成。

后续原流程：`process_data_human.py` → `convert_to_pkl_human.py` → 标注／跟踪（`--process_points --smooth_tracks`）→ `convert_pkl_human_to_robot.py --smooth_robot_tracks` → `train.py agent=point_policy suite=point_policy dataloader=point_policy`。人手三维运动还需 MediaPipe 和手部到机器人的映射。仿真 Franka 记录只覆盖机器人分支，不能作为人手示范。

## 来源与检查

### 当前验证与尚未接入的功能

录制回归覆盖多段追加、三视角 RGB-D、状态／指令与时间对齐、并发写入保护、最终验证失败保留原始媒体并阻止后续追加。Pico 的佩戴／对面模式、重连、方向和共享 LeRobot 录制也有测试；本次没有连接头显，未重新做硬件验收。测试命令见 README。

保存提交已修复：最终媒体校验和元数据更新都完成后才清除 `INCOMPLETE`，失败时保留恢复文件。不自动丢帧或把未验证 episode 标为可用。

需要修复／接入的事项：

- 采集追加签名比较配置值，但没有模型资产内容及控制／评分代码版本。只改变资产或实现、不改变配置时，仍可能混合不同环境的数据；应保存并比较相关版本／摘要。已有成功条件配置值在 signature 中，不能说完全没有校验评分配置。
- 米制夹爪反馈／目标和七关节下发目标尚未接入；需同步升级 raw、LeRobot、算法导出、推理反变换和 schema，不能只改字段名，也不能把实测 qvel 当速度指令。
- 当前 raw 的 `command_7d` 来自已处理的 `row['action']`；未独立记录原始输入请求及每次裁剪／重锚事件。虽然配置和实际指令已保留，仍不足以逐次还原输入被改变了多少，应补全审计字段。
- ViSk 当前 6 Hz 与论文 10 Hz 的时序差异仍未解决；需联合检查标签时间间隔和部署时钟。线材 ViSk 闭环仍未接入，新三相机策略未训练。FTF 视觉跟踪与完整闭环未验证。
- Robotiq 喂食已接入源连杆初始关节、实际指腹中点放勺和 EBiM 接触预紧初始化，保留三路相机；连续夹爪目标仍按实际下发值保存。预紧只用于 reset，不钳制后续策略／遥操作指令；再次全闭合与复杂线材接触的稳定性仍未解决。cube／喂食的 Panda 默认 Menagerie 伺服，线材的 Panda 默认 robosuite 伺服，Robotiq 保留 EBiM 转轴伺服，不能声称所有任务控制器参数相同。
- 当前默认拟合偶极子的幅值系数来自旧网格和 Pollen 伪标签分布；48 点降阶网格、方块和勺子的响应仍缺独立验证，不能称为真机标定。材料／接触数值参数及网络局限见[模型与标定](MODEL_CALIBRATION_ZH.md)。

控制限制的开启／关闭影响也记录在该文档；删除局部高度截断、冻结／补零不代表取消全部限制。后续修改应保留原始反馈、原始请求和实际下发命令，明确记录裁剪／重锚事件；旧 schema 不与新数据静默混用。

### π0.5 / NeoSim：采集协议核对（2026-10-11）

以下是源码核对与后续接入要求，**没有修改当前录制协议，也没有启动训练**。π0.5 是策略模型，不规定所有机器人必须使用同一种夹爪单位或动作空间。

- **OpenPI 的 DROID 自定义数据示例**：原始 `trajectory.h5`＋各相机 MP4＋语言注释，经转换写成 LeRobot。示例保存两路外部 RGB 与一路腕部 RGB、7 维实测关节位置、1 维夹爪状态、8 维动作及任务文字，15 Hz；图像 feature 声明为 `image`，不能把它描述成必须使用 MP4 的 LeRobot 写法。动作来自 `action/joint_velocity`＋`action/gripper_position`，不是实测关节速度。固定版本：[转换脚本](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/convert_droid_data_to_lerobot.py)。
- **DROID 夹爪是归一化宽度**：`g = 1 - width / max_width`，0 开、1 闭；下发时转换回 `target_width = max_width * (1-g)`。[DROID 驱动](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py)。本项目当前是驱动关节行程比例，尤其 Robotiq 转轴角度的比例不能直接声称等于 DROID 的宽度比例。
- **保存三视角不等于模型使用三视角**：原 π0.5-DROID 输入适配实际启用一路外部 RGB＋腕部 RGB，以及关节／夹爪状态和语言；不读取 AnySkin 15 路磁场或深度。[输入适配](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py)。部署示例还将预测的夹爪目标按 0.5 二值化；这属于其部署代码，不是 LeRobot 格式要求，不应在本项目中隐式照搬。[部署示例](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/main.py)。
- **仿真可采用官方关节位置训练分支**：OpenPI 的完整 DROID 微调示例使用 RLDS，动作取下发的关节位置目标，训练时前七维转为相对关节位置，输出时还原；夹爪保持绝对目标。该分支与预训练 π0.5-DROID 的关节速度接口不同，不能直接替换原 checkpoint 的执行语义。自有小数据集用 LeRobot 时，需要明确配置同等的动作转换，而非直接运行默认速度版配置。[训练说明](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README_train.md)、[配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py)。
- **NeoSim 原始数据**：公开代码位于 `neoteai/N0-Foundation/neosim`，核对版本 `ad2d9b1ead7274391b348641e9abf38167e04f1e`。脚本策略＋cuRobo 按 seed 自动采集；每条演示保存 `hdf5/<seed>.hdf5`，另外保存拼接预览 `video/<seed>.mp4`、metadata 和成功记录。HDF5 含相机、机器人关节／末端位姿、触觉等；RGB 在 HDF5 中编码为 JPEG 字节，预览视频不是唯一图像来源。[采集说明](https://github.com/neoteai/N0-Foundation/blob/ad2d9b1ead7274391b348641e9abf38167e04f1e/neosim/docs/Collection.md)、[保存实现](https://github.com/neoteai/N0-Foundation/blob/ad2d9b1ead7274391b348641e9abf38167e04f1e/neosim/envs/utils/data.py)。
- **NeoSim 夹爪与标签**：Franka 原始 joint 是七个臂关节＋两个手指关节，单臂 ACT 转换选前八维；最后一维是一个手指的直线关节位置（米），不是两指总间距。ACT 转换取 `q[t]` 作 state、`q[t+1]` 作 action。π0.5 单臂客户端也使用七关节＋一个手指位置，下发关节位置目标。不能把该 ACT 标签做法当成 OpenPI 的指令记录方法，或直接替代本项目的夹持指令。[ACT 转换](https://github.com/neoteai/N0-Foundation/blob/ad2d9b1ead7274391b348641e9abf38167e04f1e/neosim/policy/ACT/process_data.py)、[单臂部署](https://github.com/neoteai/N0-Foundation/blob/ad2d9b1ead7274391b348641e9abf38167e04f1e/neosim/eval/eval_pi05_single_joint_cl.py)。

当前接入缺口：本项目已保存三相机、instruction、实测七关节位置／速度、TCP 状态／指令、夹爪状态／指令、深度和磁场；**没有独立记录实际下发的七关节位置目标／DROID 语义的速度指令，也没有保存与真机驱动同义的夹爪反馈宽度与目标宽度（米）**。宽度约定见本文开头，不要求硅胶接触面间距。当前 7 维 TCP `action` 不能改个字段名就作为 DROID 的 8 维关节动作。建议保留公共 LeRobot＋100 Hz raw，补全实际关节命令与驱动反馈宽度，再分别导出 ViSk、FTF、OpenPI；旧数据不伪造缺失命令。这些额外字段尚未接入；当前 cube / ViSk 流程仍使用上述 TCP 与连续夹爪指令，未偷偷更换协议。

固定参考源码：

- [ViSk 项目](https://visuoskin.github.io/)；本地 `third_party/official/visuoskin`，提交 `66519d2ffc09df7df33f1f0807c88f29f849be23`。
- [FTF 项目](https://feel-the-force-ftf.github.io/)；本地 `third_party/official/feel-the-force`，提交 `954830f94e59e47c1668ad41c36b52626da975d0`；保留子模块固定版本。

```bash
.venv-reduced/bin/python -m wowskin_lab official status
.venv-reduced/bin/python -m wowskin_lab official verify data/exports/visk_cube/workspace
.venv-reduced/bin/python -m wowskin_lab dataset-check data/episodes/pick_cube/lerobot
```

W&B 可在以后训练时独立同步指标：`scripts/watch_visk_wandb.py TRAINING_DIR --online --project WowSkin-Lab --entity YOUR_WORKSPACE`，只同步 loss、步数、耗时和配置，不上传视频／数据／权重。

物理局限、本地控制参数及 Robotiq 强夹持故障见 [模型与标定](MODEL_CALIBRATION_ZH.md)。采集接口测试通过不代表触觉已真实标定，也不代表两篇论文的完整成绩已复现。
