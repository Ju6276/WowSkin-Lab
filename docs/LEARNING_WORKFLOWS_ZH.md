# ViSk / FTF 学习工作流

更新：2026-10-09。目标是执行官方完整方法：ViSk 按论文单块 AnySkin/15 维；FTF 保留机器人分支和人手主线。代码模型、损失、归一化、动作块不使用历史简化补丁，物体点由视觉跟踪获得。

## 当前进度与 Mac 支持

| 环节 | 本机状态 |
| --- | --- |
| 仿真遥操作、自动采集、视频/状态/触觉同步 | 已运行；两段自动抓起/放回成功 |
| ViSk 原版 PKL / 加载器 / CPU 训练 | 已完成一次更新，loss 14.711，checkpoint 已保存；退出时资源回收卡住，经清理退出，兼容问题尚未修复 |
| FTF 原版视频对齐、初始 PKL | 已运行；初始文件还不能训练 |
| FTF 官方点标注 notebook | 已配置并通过执行检查；**还未实际点击并保存标注点** |
| FTF DIFT＋CoTracker 完整跟踪 | 未运行；固定版调用 CUDA/xformers，当前 Mac 不能原样执行 |
| FTF 最终数据训练、两种策略闭环 | 未验证；不能将缺少数据等同于算法不能在 CPU 运行 |
| FTF 人手采集、真实 sensor 标定 | 未完成 |

这些是 Python 项目，不按“有没有 Mac 安装包”划分。Notebook/预处理已在 Mac 验证；阻塞完整 FTF 的是当前视觉流水线的 CUDA 依赖。单独组件的 CPU/MPS 移植尚未验证，不替换模型来冒充原版。用户暂时没有 CUDA 机器，先完成本机部分。

日常操作见 [遥操作与采集](PICK_CUBE_ZH.md)。ViSk 本机训练产物保存在 `outputs/diagnostics/visk_original_cpu_20261009/`；目前不宣称论文成功率或完整复现。

## 源码与环境

- ViSk：`third_party/official/visuoskin`，`66519d2ffc09df7df33f1f0807c88f29f849be23`。
- FTF：`third_party/official/feel-the-force`，`954830f94e59e47c1668ad41c36b52626da975d0`。
- FTF 内的 Franka-Teach、DIFT、CoTracker 均保留主仓库记录的子模块版本，没有切换到最新 main。

```bash
.venv/bin/python scripts/setup_upstream.py
.venv/bin/python -m wowskin_lab official status
```

官方环境分别以 ViSk `env.yml`（Python 3.11 / PyTorch 2.1 / CUDA 11.8）和 FTF `conda_env.yaml`（Python 3.10 / CUDA 依赖）为准。当前 Mac 是 Apple Silicon，已确认 `torch.cuda.is_available() == False`。FTF 的 `correspondence.py` 直接调用 `.cuda()`，DIFT 启用 xformers；不能把 CPU 简化替代说成官方流程已完成。

## 三条路线

**ViSk**：原始 XArm 数据用 `process_xarm_data.py`；本项目 MuJoCo 数据用明确记录的采集格式桥接器，两者随后都进入原版 `convert_to_pkl.py` → `train_bc.py`。README 的 `process_data.py` 与实际文件名不同，以代码为准。保持 batch 256、动作块 10、subsample 5、训练 10000 步；部署 `eval.py` 仍是 XArm 接口，尚未接入 MuJoCo 闭环。

[论文 IV-A](https://arxiv.org/html/2410.17246v2#S4.SS1) 明确是左 AnySkin、右普通硅胶。公开默认代码却读两组磁场，作者未解释原因。按用户要求，`official prepare visk` 默认 `paper-single`：只把加载器数量、环境数量改为 1，配置改为 `sensor0`，共三行；生成 `PAPER_SINGLE_SENSOR.patch` 和 manifest。官方参考 checkout 保持原样，模型、损失、归一化（含最小尺度 100）、基线处理、动作块和采样逻辑不改。不补零、不复制第二组。

**FTF 机器人分支**：机器人遥操作记录 → `process_data_robot.py` → `convert_to_pkl_robot.py` 初始转换 → `label_points.ipynb` 标注 → 再运行转换器并启用 `--process_points` → `train.py agent=p3po suite=p3po dataloader=p3po` → 官方带接触反馈的推理。沿用官方 README 的 FTF 机器人设置：subsample 15、`binarize_gripper_data=true`；batch 64、历史 10、动作块 20、训练 100100 步来自固定版配置。

**FTF 人手主线**：人戴触觉手套直接操作物体 → `process_data_human.py` → `convert_to_pkl_human.py` → 标注物体点 → 转换器 `--process_points --smooth_tracks` → `convert_pkl_human_to_robot.py --smooth_robot_tracks` → `train.py agent=point_policy suite=point_policy dataloader=point_policy` → 官方带接触反馈的机器人推理。使用官方 subsample 3；batch 64、历史 10、动作块 20、训练 100100 步。

FTF 两条路线都保留 DIFT / CoTracker 物体跟踪；人手主线额外保留 MediaPipe 和手部映射。机器人九个参考点按官方位姿几何构造；物体点来自任务的人工语义标注和视觉跟踪，不固定为八个方块角点。

上述数字是固定版本公开配置，并非断言所有论文实验使用同一训练步数。模型、归一化和控制器不应用此前历史补丁；数据加载器只有上面明确记录的传感器数量变更。公开代码中的错误/论文差异需逐项记录。

## 采集后如何处理

所有命令从 项目根目录 运行。选择 `both-raw` 同时录两种格式；操作、频率、字段和起止规则统一见 [采集说明](PICK_CUBE_ZH.md)。

线材场景已接入 `teleop --task cable --record-format both-raw --ideal`，使用形变＋磁偶极子输出，数据单独写入 `visk/raw/cable` 与 `ftf-robot/extracted_data/cable`。本页已有转换命令和工作目录仍针对 `pick_cube`，不能直接当作线材配置使用；后续须准备任务名 `cable` 的独立工作目录，FTF 物体点依据线材任务视觉标注确定。已接通原始录制，不表示线材策略已训练或布线成功率已验证。

**ViSk**：已准备 `outputs/official/visk-paper-single` 对应默认录制目录。

```bash
.venv/bin/python -m wowskin_lab official process-visk-sim \
  --workspace outputs/official/visk-paper-single
```

输出 `data/official/visk/processed_data_pkl_aa/pick_cube.pkl`，并检查原版加载器输入：四路 `[1,3,128,128]` 图像、`sensor0=[1,15]`、`actions=[1,10,7]`。原版转换器负责 BGR 色序、腕相机裁剪、128px 缩放和统计量。原 XArm 预处理依靠位姿完全相等推断暂停；仿真桥接仅改用录制起止事件切段，不伪造静止位姿。已扣/未扣基线不能混用。

现有处理结果不覆盖。新批次可用新的 `data-root/workspace`，通过 `--raw-root` 指向原始目录。训练命令在对应工作目录的 `COMMANDS.md`；Mac 支持范围见本文开头。PickCube 是本项目任务，方向噪声、实物任务和成绩尚未复现。

**FTF 机器人分支**：先对齐，再产生供标注的初始图像 PKL。

```bash
.venv/bin/python -m wowskin_lab official process-ftf-robot \
  --data-root data/official/ftf-robot --task pick_cube
.venv/bin/python -m wowskin_lab official convert-ftf-robot-initial \
  --data-root data/official/ftf-robot --task pick_cube \
  --calib-path data/official/ftf-robot/extracted_data/pick_cube/demonstration_0/sim_camera_calibration.npy
```

第一步运行原版 `process_data_robot.py` 生成 MP4/CSV；第二步运行原版 `convert_to_pkl_robot.py`，不加 `--process_points`。输出位于 `processed_data_pkl/expert_demos/franka_env/pick_cube.pkl`，**目录叫 expert_demos 也不代表已能训练**；同名 `.initial_conversion.json` 记录 `training_ready=false`。包装入口只补建目录、检查长度/有限值、阻止追加。

## FTF 点标注与自动跟踪

这是两个独立步骤：**Notebook 人工选起点 → Python 脚本自动跟踪视频 → 检查轨迹 → 最终 PKL**。机器人分支九个参考点使用官方位姿几何；物体点数由语义标注决定，不固定为八个方块角点。

### 第一步：在 Mac 打开 notebook 标注

本机已有两段自动采集示范，以下文件都真实存在：

- [对齐后双目视频和状态](../data/automated/flow_check_20261009/ftf-robot/processed_data/pick_cube)。
- [初始 PKL](../data/automated/flow_check_20261009/ftf-robot/processed_data_pkl/expert_demos/franka_env/pick_cube.pkl)，约 944 MiB，因为包含两路完整图像序列。
- [官方标注 notebook 工作副本](../outputs/official/ftf-annotation-20261009/label_points.ipynb)，只填了任务与路径，原始算法单元未改。

```bash
cd WowSkin-Lab
.venv/bin/jupyter notebook outputs/official/ftf-annotation-20261009/label_points.ipynb
```

1. 顺序运行单元，保持 `traj_idx=0`、`original_bgr=True`、`size_multiplier=1`。
2. 最后一个单元先设 `pixel_key="pixels1"`，在图上点击语义明确的物体点，按 Save Points。
3. 改为 `pixel_key="pixels2"`，重新运行最后一个单元，按**同一物理点、同一顺序**点击并保存。两侧都必须可辨认，不能把遮挡位置随意猜成对应点。
4. 保存到 `data/automated/flow_check_20261009/ftf-robot/coordinates/pick_cube/`：`images/pixels1.png`、`pixels2.png`，以及 `coords/pixels1_objects.pkl`、`pixels2_objects.pkl`。坐标为 `[(0,x,y), ...]`，0 表示首帧。原注释建议每视角最多 10 点；点数必须与任务配置一致。

通常每个任务标一个参考示范的首帧，不必逐帧手点。当前只检查了 notebook 能执行，**没有完成上述人工标注**。新数据可用 `official prepare-ftf-annotation --data-root ROOT --task TASK --output NEW_DIR` 生成副本；依赖在 `requirements-annotation.txt`。

### 第二步：在 CUDA 环境自动跟踪

准备原版权重、任务配置、双目标定和 TCP 配准，将标注目录放入官方工作目录 `source/coordinates/pick_cube/`。运行该工作目录 `COMMANDS.md` 的 `track` 命令，即原版转换器加 `--process_points`。

- **DIFT**：将参考图上的点匹配到各段示范首帧。
- **CoTracker**：跟踪这些点在后续帧中的二维位置。
- **双目三角化**：用内外参恢复三维物体轨迹，检查跟踪视频后再训练。

当前固定版 [correspondence.py](../third_party/official/feel-the-force/point_policy/point_utils/correspondence.py) 调用 `.cuda()`，DIFT 启用 xformers。不能只把 notebook 打开或配置 `device=cpu` 就认为整条流水线可在 Mac 运行。[原版 PointsClass](../third_party/official/feel-the-force/point_policy/point_utils/points_class.py)

**运行跟踪前先归档同路径的初始 PKL**：原版检测到已有文件会追加，否则会混入未跟踪记录。人手主线还需要 MediaPipe 和 `convert_pkl_human_to_robot.py`，机器人示教不替代此步骤。

截至本次核对，官方未找到可直接下载的完整 FTF 同步示教数据包/预训练任务策略；官网展示视频不能替代触觉、标定和时间戳。CoTracker 自带 [apple 示例视频](../third_party/official/feel-the-force/co-tracker/gradio_demo/videos/apple.mp4)可看组件格式，但不是 FTF 示教。

## FTF 方法与分支

FTF 用历史点轨迹与接触状态，经 MLP/Transformer 行为克隆预测未来运动和目标接触强度。执行时机器人跟踪运动，夹爪根据目标与实测触觉误差调节闭合。原版机器人转换器使用起始五帧均值作基线，取首个磁力计 XYZ 差的模长；不是三维牛顿力回归。[论文 §3](https://arxiv.org/html/2506.01944v1#S3)

论文称接触反馈为 PD 控制，固定版 `force_step` 可见按误差比例更新；增益与信号尺度/频率相关，当前尚未移植并验证。未使用的 `sensor_history` 解析硬编码为 30 通道，输出接触量只取前三路；这些公开实现差异仍按原样记录。原版源码公开，固定版本根目录未找到 LICENSE，不替作者宣称 MIT 等许可。

## FTF 人类视频到底怎么录

[论文 §3.1–3.2](https://arxiv.org/html/2506.01944v1#S3) 中，人佩戴拇指掌侧带 AnySkin 的手套，直接完成抓取等任务；两台固定、标定过的相机拍摄手和物体，同时经 USB 记录触觉。论文描述视频 30 fps、触觉原始 200 fps，再按时间对齐。视频不是后期展示录像，也不能替代同步触觉。

**人手和机器人两条路线共用 Franka-Teach 的记录器，不只用于机器人遥操作。** 本地源码：

- [teleop.py](../third_party/official/feel-the-force/Franka-Teach/teleop.py)：`teleop_mode=human` / `robot` 分支。
- [teleoperator.py](../third_party/official/feel-the-force/Franka-Teach/frankateach/teleoperator.py)：human 模式采集期间不把手柄位姿发给机器人运动控制；按钮 A/B 更新 `start_teleop`，用于后处理截取示范。启动时仍有机器人 reset/offset 调用，并持续请求机器人状态，因此公开入口依赖机器人服务，不能称为独立录像工具。
- [camera_server.py](../third_party/official/feel-the-force/Franka-Teach/camera_server.py)：发布双相机图像；官方 camera.yaml 为 640×480、30 fps。
- [reskin_server.py](../third_party/official/feel-the-force/Franka-Teach/reskin_server.py)：固定子模块实际的触觉服务入口（主仓库 README 写成 reskin.py）。
- [collect_data.py](../third_party/official/feel-the-force/Franka-Teach/collect_data.py) → [DataCollector](../third_party/official/feel-the-force/Franka-Teach/frankateach/data_collector.py)：接收图像、状态、触觉并存盘。

在官方服务、VR、相机和串口均配置完成后，其人手采集命令是：

```bash
# 在 Franka-Teach 目录，两个终端分别运行；这些服务尚未在本机接通。
python teleop.py teleop_mode=human
python collect_data.py collect_img=true collect_state=true collect_reskin=true demo_num=0
```

每段输出 `cam_1_rgb_video.avi`、`cam_2_rgb_video.avi`，各自的 `.metadata` 保存时间戳；`reskin_sensor_values.h5` 保存 `timestamp`、`sensor_values`、`sensor_history`；另有 `states.pkl`、`commanded_states.pkl`，状态包含示范起止标记。`DataCollector` 的写入从进程启动持续到停止，VR 起止标志供预处理裁剪，不等于按一下按钮就关闭视频文件。

之后：MediaPipe 从两路视频检测手部关键点，双目三角化获得三维位置；物体点经人工初始标注、DIFT 对应和 CoTracker 跟踪；最后将手指运动映射为机器人参考点并与触觉对齐。输入是**视频＋同步触觉＋时间/标定信息**，训练 PKL 是这些处理的产物。

已发现的公开版本差异：论文写触觉 200 fps，但固定 Franka-Teach 触觉发布器使用 `FrequencyTimer(100)`；README 的 `num_mags=5` 示例与子模块配置默认 10 不同。需要依据实际接入通道和时间戳核对，不能隐藏这些差异。

## 准备可审查的官方工作目录

`official prepare` 复制固定版本上游代码，填写路径/任务配置；ViSk 默认另外应用上面三行论文单传感器配置，逐文件记录 SHA256 和变更理由。不会训练、打开机器人连接、编造数据或自动执行步骤。

```bash
.venv/bin/python -m wowskin_lab official prepare visk   --workspace outputs/official/visk --data-root data/official/visk --task pick_cube

# pick_bread 是官方配置文件名；其真实数据名由 YAML 读取，不被擅自改名。
.venv/bin/python -m wowskin_lab official prepare ftf-robot   --workspace outputs/official/ftf-robot --data-root data/official/ftf-robot --task pick_bread
.venv/bin/python -m wowskin_lab official prepare ftf-human   --workspace outputs/official/ftf-human --data-root data/official/ftf-human --task pick_bread
```

这些是**配置准备示例，数据尚不存在，不表示这些任务已复现**。自建任务使用 `--task-config PATH.yaml`，物体点数必须与实际语义标注一致；没有自动把 PickCube 当成官方任务。

每个工作目录的 `COMMANDS.md` 列出采集/预处理/训练/推理命令及前置条件；`manifest.json` 记录来源。推理命令仍是官方真实机器人接口，MuJoCo 后端尚未接通。检查是否出现意外源码修改：

```bash
.venv/bin/python -m wowskin_lab official verify outputs/official/ftf-robot
```
