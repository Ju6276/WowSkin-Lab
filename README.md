# WowSkin-Lab

Franka 触觉遥操作、RGB-D 数据采集和 ViSk / FTF 算法接入。基于 MuJoCo，支持 Panda / Robotiq 2F-85、键盘 / SpaceMouse / PICO 4，提供方块搬运、线材插扣、喂食、Peg Insert，以及实验性薯片夹取场景。

建议从 Panda 方块任务开始。ViSk 提供数据转换、训练与仿真闭环入口；FTF 提供数据导出和标注入口，后续视觉跟踪需要 CUDA。软指腹与磁场是未完成真机标定的本地近似模型，本项目包含数据与环境适配。薯片夹持仍待调整，保留为实验场景；已知问题统一见 [TODO](TODO.md)。

[启动与录制](#启动与录制) · [其他任务](#其他任务与输入) · [数据与学习](#数据与学习) · [代码与检查](#代码入口与维护)

## 安装

当前安装入口面向 Apple Silicon macOS、Python 3.12（与 `.python-version` 一致）、MuJoCo 3.15，从源码目录运行。`pyproject.toml` 声明最低 Python 3.10；这不表示完整安装流程已在所有声明版本上验证。Linux NVIDIA 的可选 GPU 物理及采集入口见 [GPU 采集说明](docs/GPU_COLLECTION_ZH.md)，兼容设置和已验收范围须单独核对。

```bash
git clone https://github.com/Ju6276/WowSkin-Lab.git
cd WowSkin-Lab
python3.12 scripts/setup_capture.py
.venv/bin/python scripts/setup_upstream.py --only franka
```

只使用一个 `.venv`，采集、遥操作、ViSk 训练和部署共用。安装入口是 `scripts/setup_capture.py`，根目录 `requirements.txt` 包含运行与学习依赖；FTF notebook 的 `requirements/annotation.txt` 和下述可选 TACTO 依赖按需加装到同一个环境。LeRobot 仅安装所用数据 API 及依赖，不是其全套机器人运行环境。

`pip check` 会报告 LeRobot 完整包所需但本项目未安装的八项依赖：`diffusers`、`cmake`、`opencv-python-headless`、`pynput`、`torchcodec`、`gymnasium`、`rerun-sdk`、`flask`。当前录制／读取使用 OpenCV 与 PyAV；完整 LeRobot 策略或硬件 API 需要另行补齐依赖。

## 启动与录制

从项目根目录运行，一次只开一个遥操作窗口：

```bash
# 无外设：键盘，保存到 data/episodes_keyboard/pick_cube/
.venv/bin/mjpython -m wowskin_lab teleop --task pick_cube --input keyboard --ideal --record-dir data/episodes_keyboard
# SpaceMouse：默认追加到 data/episodes/pick_cube/
.venv/bin/mjpython -m wowskin_lab teleop --task pick_cube --input spacemouse --ideal
```

启动后先测试操作，**不会自动录制**。SpaceMouse 左侧键闭合、右侧键张开；键盘 W/S 前后、A/D 左右、E/Q 上下，I/K、L/J、U/O 旋转，X/Z 闭合／张开、C 保持。点击仿真窗口后操作。

三种输入默认使用固定机器人基座方向，转动观察视角不改变输入映射：W/S 对应基座 ±X，A/D 对应 −Y/+Y，E/Q 对应 ±Z。键盘旋转键仍绕 TCP 局部轴。**本地桌面工作空间盒子默认关闭**；速度、目标领先量、输入死区与底层控制参数仍保留，详见[控制说明](docs/MODEL_CALIBRATION_ZH.md#生效的控制与质量检查)。`--control-frame view` 和 `--workspace-limits` 分别显式恢复旧视角映射与旧边界。新旧控制配置请使用不同录制目录。

两项**默认关闭**的接触对照：`--pad-backing-contact` 恢复硅胶与硬底座的碰撞支撑，但目前明显增加计算成本，强夹薄线仍可能翻转；`--finger-contact-stop` 在两块指腹互碰时保持当时开合目标，张开后解除。后者是操作选项，不是物理模型修复。启用后界面明确显示状态，使用新录制目录。默认模型的大压缩问题仍未解决，见 [TODO](TODO.md)。

1. **B**：复位，看到红色 `REC capture_…` 后开始操作。
2. 完成任务，左上角绿色 **SUCCESS** 提示自动达标（薯片试验无自动成功判定）。
3. **Enter**：结束并标记成功，等待 `SAVING → VERIFYING → SAVED`。
4. 显示 **SAVED** 后，按 **B** 开始下一条；编号自动递增，重启也能追加。

**N** 保存失败，**R** 中断并复位，**P** 保存中断记录并退出。自动达标不会结束录制；人工标签与自动判定分别保存，`SAVED` 只表示写入完成。**H** 展开诊断，**V** 切主视角。不同夹爪／相机／控制配置使用新的 `--record-dir`。

## 其他任务与输入

```bash
# 单卡扣插线；Panda，默认刚性卡扣
.venv/bin/mjpython -m wowskin_lab teleop --task cable --cable-layout single_clip --input spacemouse --ideal
# 双连接座长线＋两柱绕线；约 0.756 m、48 段，两端固定
.venv/bin/mjpython -m wowskin_lab teleop --task cable --cable-layout routing --input spacemouse --ideal --record-dir data/episodes_routing_v1
# SERL++ Peg 素材：启动已物理夹持，插入孔座
.venv/bin/mjpython -m wowskin_lab teleop --task peg_insert --input spacemouse --ideal
# Panda 喂食：初始化已夹勺，并放好六颗豆子
.venv/bin/mjpython -m wowskin_lab teleop --task feeding --input spacemouse --ideal
# Robotiq 喂食，独立数据目录
.venv/bin/mjpython -m wowskin_lab teleop --task feeding --gripper robotiq_2f85 --input spacemouse --ideal --record-dir data/episodes_robotiq
# PICO 4：USB 调试已授权，adb 在 PATH 或 .tools/platform-tools/
.venv/bin/mjpython -m wowskin_lab teleop --task pick_cube --input pico --ideal --record-dir data/episodes_pico
```

`single_clip` 仍为 0.38 m、24 段、首端固定，按 `local_cable_clip_v2` 检查入槽。`routing` 为独立双端连接布局，按顺序绕过两个柱的指定侧面评分；卡扣入槽不替代绕柱成功。两者的透明引导框只显示在操作窗口。历史记录保持原定义，新版使用新录制目录。[布局和评分说明](docs/COLLECTION_ZH.md)。

Cube、线材、喂食、Peg 均可加 `--gripper robotiq_2f85`，默认 Panda；两种夹爪都保留连续反馈与连续指令。Peg 使用原始 SERL++ 网格与凸分解，依靠接触夹持，不引入上游的 peg 焊接约束；本地抓法、评分与控制差异见采集手册。

**Robotiq 强闭合仍有已知问题**：薄线夹持或空夹时，完整 0.8 rad 指令可能压翻新增软指腹，中止该次无效采集。没有截短行程来规避；`tests.physics.test_robotiq` 保留失败复现。Robotiq 喂食的初始化夹持与完成录制是另一种负载条件，不能据此认为所有任务已稳定。

机械臂默认仍为 IK＋关节 PD。可选笛卡尔阻抗和实验弹性卡扣分别启用，不会自动替换：

```bash
.venv/bin/mjpython -m wowskin_lab teleop --task cable --cable-layout single_clip --input keyboard --arm-controller cartesian_impedance --clip-model elastic_beam --ideal --record-dir data/episodes_elastic_cartesian
# Robotiq 版本再加 --gripper robotiq_2f85，并另选录制目录
```

笛卡尔阻抗使用固定版 SERL++ 控制函数，reset 仍通过原关节 PD 建立初始夹持。弹性卡扣的上下唇均可弯曲，安装底座固定；这是本地平面离散梁近似，材料与插拔力未标定。默认卡扣仍为刚体；两种线材布局均支持显式选择弹性版。实验范围见[模型说明](docs/MODEL_CALIBRATION_ZH.md#卡扣的物理范围)。

Pico 支持佩戴／对面模式，都需在头显浏览器中进入 WebXR；松开右握键一次后，按住跟随。完整按键、成功条件和故障处理见 [采集手册](docs/COLLECTION_ZH.md)。Robotiq 强夹持、线材压扣仍可能出现软指腹网格失效，不承诺任意接触稳定。

右侧显示两路固定相机＋一路腕部相机。加 `--pointcloud` 可在其下预览 RGB-D 融合点云；逐帧外参将腕部视角变换到机器人基座系。使用 MuJoCo 理想深度，目前不模拟深度噪声。预览不写进相机图像；快捷键与开销见采集手册。

## 可选视触觉传感器

每次只选一种传感器，左右各装一个同类型模块。默认 AnySkin；DIGIT 可选 TACTO 或 FOTS，GelSight 使用独立的 GelSightMujoco 原模型（不是 Mini）。三种光学配置均接入 Panda 五类任务、三路场景 RGB-D、双指光学显示和手动录制；Robotiq 光学安装尚未接入。

TACTO 的场景长指部现在使用不透明黑色；旧录像保留采集时的颜色。六条黑色版 TACTO 示例已重录，见[示例记录与本机使用说明](examples/optical/README.md)。继续采集请使用新录制目录。

```bash
# 可选依赖装到同一个 .venv，固定源代码／CAD／权重版本
.venv/bin/python scripts/setup_upstream.py --only tacto fots_mujoco gelsight_mujoco
.venv/bin/python -m pip install -r requirements/tacto.txt
.venv/bin/python -m pip install --no-deps -e third_party/mujoco-tacto

# 三选一；--task 可换 cable / feeding / peg_insert / chip_tray
.venv/bin/mjpython -m wowskin_lab teleop --sensor digit --optics tacto --task pick_cube --input keyboard
.venv/bin/mjpython -m wowskin_lab teleop --sensor digit --optics fots --task pick_cube --input keyboard
.venv/bin/mjpython -m wowskin_lab teleop --sensor gelsight --task pick_cube --input keyboard

# 独立按压台：球／方体／圆柱探头；两套 DIGIT 各自原模型的比较
.venv/bin/mjpython -m wowskin_lab sensor --sensor digit --optics fots --probe-shape box
.venv/bin/python -m wowskin_lab digit-compare --ideal
```

任务窗口操作与上文相同，**B 开始、Enter 保存**。光学图位于左下，三路相机位于右侧；逗号 `,` 切换深度／接触区。**ROBOT LEFT / RIGHT 按机器人手掌 +Y / −Y 定义，不随观察视角交换**。`--ideal` 关闭 TACTO 随机 RGB 噪声或 GelSight 纹理噪声，不删除原背景；FOTS 不额外添加随机噪声。

TACTO／GelSight 完整长组件替换原短指几何；FOTS 独立模块装在原指尖。保留各自源 CAD、接触参数和光学函数，安装变换与任务控制属于本地适配，详见[模型与来源](docs/MODEL_CALIBRATION_ZH.md#传感器硬件与光学算法)。没有用 AnySkin 参数代替光学传感器。

光学数据使用独立 schema：100 Hz 机器人／接触力，约 30 Hz 场景 RGB-D 和双指光学帧；不填充 AnySkin 15 路字段。三个默认采集根目录分别为 `data/episodes_tacto/`、`data/episodes_digit_fots/`、`data/episodes_gelsight/`，录制器会再追加任务名。例如方块成功数据位于 `data/episodes_tacto/pick_cube/lerobot/`；自定义 `--record-dir` 也遵循此规则。数据字段、独立台按键与验证命令见[光学采集说明](docs/COLLECTION_ZH.md#光学触觉数据)。

**使用边界：** FOTS 多刚体接触的 marker 不受所接入源代码支持；GelSight 大平面覆盖整个视野时 RGB 响应可能很弱。原始深度定义不同，不可直接比较为真实压缩或精度。现有 ViSk／FTF 磁场网络不能直接训练光学图像；各任务完整操作成功率与实物标定仍待验证，见 [TODO](TODO.md#独立视触觉传感器与后续应用)。

已新增 `dataset-review` 离线多模态回放和 `optical-act train|eval` 本地光学基线，调用 Hugging Face LeRobot 0.3.3 ACT；命令、输入与验收边界见[光学学习说明](docs/OPTICAL_LEARNING_ZH.md)。源代码与实际 UniVTAC 数据／图像对照见[光学数据检查记录](docs/OPTICAL_DATA_AUDIT_ZH.md)。当前三个光学来源并非 Isaac GelSight Mini 的等价实现。

## 脚本自动采集

统一入口是 `python -m wowskin_lab record-auto`；参数解析和方块脚本在 [auto_collect.py](wowskin_lab/data/auto_collect.py)，喂食／single clip 控制器在 [scripted_tasks.py](wowskin_lab/data/scripted_tasks.py)。脚本读取仿真物体姿态，通过正常控制接口执行，不是 agent 推理或人工遥操作。

```bash
# 喂食：从初始化已夹持、装豆的勺子开始，送至嘴前区域
.venv/bin/python -m wowskin_lab record-auto --task feeding --sensor-backend tacto \
  --episodes 1 --output data/automated/tacto_feeding
# 单卡扣：显式启用上下唇弹性，夹持线材并送入槽内
.venv/bin/python -m wowskin_lab record-auto --task cable --cable-layout single_clip \
  --clip-model elastic_beam --sensor-backend tacto --episodes 1 \
  --output data/automated/tacto_single_clip_elastic
```

将 `--sensor-backend tacto` 换成 `digit_fots` 或 `gelsight` 可选择另外两种光学后端，同时换用新的输出目录。此入口用 `--output` 指定**尚不存在**的目录；成功数据分别保存在其下的 `feeding/lerobot/` 和 `cable_insert/lerobot/`，对应 `raw/` 保留原始数值与元数据，根目录保存 `collection_report.json`。默认录制三路场景 RGB／深度、双指光学图像与数值、机器人状态及实际控制指令。方块使用 `--task pick_cube`（默认）；`--randomize` 当前仅支持方块。

`--clip-model elastic_beam` 使用已有的上下唇被动弹性模型，安装底座固定；参数是未标定的平面梁近似。省略该参数或选择 `rigid` 则使用刚性卡扣，必须另用输出目录。三种光学后端使用各自的脚本抓取／推进参数，记录在 `collection_report.json` 和 raw 元数据中。两种卡扣都按中心线入槽并保持评分，不能据此推定松手后锁紧；喂食评分不包含吞咽。此前十五条回放中的 single clip 为刚性版，新弹性记录与之分开。

```bash
.venv/bin/python -m wowskin_lab dataset-check data/automated/tacto_feeding/feeding/lerobot
.venv/bin/python -m wowskin_lab dataset-review data/automated/tacto_feeding/feeding/lerobot \
  --episode 0 --output outputs/review_tacto_feeding
```

三种光学后端各一条喂食、刚性 single clip 和上下唇弹性 single clip 已完成采集、导出、读取和回放检查；单条示范不足以划分独立训练／验证集，这两项任务尚未训练和闭环验收。本机回放入口为 `outputs/optical_replays_20261014/index.html`。十八条完整示例数据和生成的回放均保留在本机，不随仓库推送。[examples/optical](examples/optical/README.md) 只保存清单、检查结果、少量截图及本机校验／解包／重建回放说明。具体结果及字段限制见[光学学习说明](docs/OPTICAL_LEARNING_ZH.md)。

GelSight 的强化学习入口为 `python -m wowskin_lab gelsight-ddqn`：将原 GelSightMujoco DDQN 学习器接到 Panda 抓放场景，包含来源补丁、在线训练和独立种子评估。2,000 步小规模试验已跑通，但独立评估完整抓放为 **0/5**，尚未学会任务。正式实验用 `gelsight-ddqn experiment`，默认 100,000 步，支持完整断点恢复、阶段评估和最终独立测试；完成前不能视为已学会任务。安装、完整命令和实际结果见 [DDQN 说明](docs/GELSIGHT_DDQN_ZH.md)。训练权重与评估视频保留在运行机器的忽略目录。

## 脆性薯片试验

**仍待调整，未验证稳定无损夹取。** 场景为康养房间、Franka／Panda、蓝绿色 EBiM 托盘和马鞍状薯片，无人体或勺子。启动后空夹爪，不自动录制：

```bash
.venv/bin/mjpython -m wowskin_lab teleop --task chip_tray --input keyboard --ideal --record-dir data/episodes_chip_tray
# 接上 SpaceMouse 后可改为 --input spacemouse
```

仅此场景的夹爪使用 MuJoCo 接触力反馈：**X** 启用（默认 0.05 N）、**Z** 打开、**C** 保持；**− / =** 每次调低／调高 0.01 N，**V** 切近景。机械臂仍用 IK＋关节 PD。显示 `INTACT / BROKEN`，没有自动 SUCCESS。该反馈不是从 AnySkin 磁场估计的力，也没有把 ViSk 的最后一维改成目标力。详细操作、独立按压台与诊断入口见[采集手册](docs/COLLECTION_ZH.md#薯片试验)，待办见 [TODO](TODO.md#薯片夹持待调整暂不作为训练主线)。

## 数据与学习

```text
data/episodes/pick_cube/
  lerobot/                   成功记录：Parquet + MP4 + meta/ + depth/*.h5
  raw/capture_000000/         metadata.json + measurements.h5
```

- **名义 30 Hz**：三路 640×480 RGB、同步米制深度、instruction、对应 state/action；另存实际仿真时间戳。
- **AnySkin 100 Hz raw**：TCP 基座系 XYZ＋rotvec、线／角速度、七关节位置／速度、夹爪反馈／目标、15 路触觉、基线及质量字段。不是 RPY。
- **夹爪 state / action 分开**：实测闭合比例作观测，实际下发闭合目标作动作；0 开、1 闭。夹物体时反馈不到 1，动作仍可为 1。另存驱动关节及单位；米制宽度尚未统一接入各任务，只有薯片 raw 的附加组已保存。
- LeRobot 的 action 是绝对 TCP／夹爪目标；ViSk 导出沿用官方位姿差分规则，得到基座系平移增量＋末端局部旋转增量＋绝对夹爪目标。两者不能直接混用。
- 成功写入并验证后，图像只留在 LeRobot，raw 不重复存图像。失败／中断保留 raw 和暂存媒体；`INCOMPLETE` 表示写入未完成，不能用于训练。

```bash
.venv/bin/python -m wowskin_lab dataset-check data/episodes/pick_cube/lerobot
```

[学习工作流](docs/LEARNING_WORKFLOWS_ZH.md)提供完整命令：**采集成功 episodes → 数据检查 → 导出／官方预处理 → 训练 → MuJoCo 闭环评估**。ViSk 使用三视角＋单块 15 路触觉，动作块 10×7，**6 Hz 策略、100 Hz 控制**；夹爪标签使用实际下发的目标。

ViSk 的学习目标保持视觉＋触觉输入，输出末端运动与夹爪开合，不输出目标力。数据中保存的本体状态用于记录和运动标签构造，当前网络输入键未包含本体状态。FTF 仍须完成官方视觉点跟踪和独立闭环接入，不能直接复用 ViSk checkpoint。

仓库保留代码、测试、依赖、配置、素材来源和操作说明，不分发本机示范、训练权重、验证报告或录像。新使用者需先采集自己的数据，再按工作流训练和评估；仅克隆代码不能复现某次训练的具体数值结果。

ViSk 训练只向 W&B 记录指标（默认项目 `WowSkin-Lab`），不生成 TensorBoard／CSV／文本训练日志。完成 3000、6000、9000 次更新分别保存 checkpoint，10000 次结束再保存最终权重。登录与训练命令见学习工作流。

## 代码入口与维护

```text
wowskin_lab/
  envs/       场景、夹爪与任务评分
  core/       接触力学、磁场与控制
  teleop/     输入命令、SpaceMouse、Pico 与目标限制
  ui/         操作窗口、触觉／相机／点云预览
  data/       录制、LeRobot、算法导出、坐标变换
  learning/   固定版本官方算法的接入层
  baselines/  Pollen、ReSkin 等可选参考模型
configs/      相机、控制、评分、传感器参数与上游版本
assets/       原始／派生素材、来源 manifest 与许可证
scripts/      安装、训练、评估；diagnostics/ 为可选实验
tests/        unit、physics、gpu、official、artifacts、manual
docs/         采集、学习、模型参数与论文索引
data/         本机示范与算法导出，不进 Git
examples/     示例清单、检查结果、截图与复现说明；大数据包不进 Git
outputs/      本机权重、标定与诊断输出，不进 Git
```

默认 AnySkin 采用 MuJoCo 3.15 降阶指腹＋拟合偶极子；光学选项使用各自源模型。`--ideal` 关闭合成触觉噪声、滤波和延迟，不关闭接触力学。控制限制、参数来源与物理失效规则见 [模型说明](docs/MODEL_CALIBRATION_ZH.md)。AnySkin 只支持当前 trilinear 降阶模型；旧完整节点模式及其选择参数已移除，不再需要 MuJoCo 3.5。旧 full 模型记录不能按降阶模型重放。

[MuJoCo GPU / CUDA Graph 后端](docs/GPU_COLLECTION_ZH.md)已接入共用环境、遥操作和自动采集：Mac 默认 CPU，NVIDIA 显式加 `--physics-backend warp`。绳子使用迁移自原插件的 Warp 弹性力，MPM 替换已撤下。GPU 兼容接触选项需显式启用；保留 CPU 初始化、IK、读数与保存，当前不承诺所有任务与 CPU 默认接触结果一致或实时。

```bash
# 以下检查包含学习接口，先按学习文档获取固定版 ViSk
.venv/bin/python -m tests unit
.venv/bin/python -m unittest tests.official.test_visk_command_labels tests.official.test_lerobot_recording
.venv/bin/python -m tests all --list
```

`unit` 检查坐标、输入和动作契约；`physics` 运行接触场景，AnySkin 物理测试不依赖 ReSkin 网络权重；`gpu` 在 NVIDIA 上对照 GPU 物理、Graph 和采集（具体未通过项见[验收记录](docs/GPU_VALIDATION_ZH.md)）；`official` 检查官方数据与学习接口；`artifacts` 依赖可选数据／权重；`manual` 需要人工操作窗口。Robotiq 强夹持的已知失败保留在物理测试中，不跳过。部分物理检查限定已验证的 MuJoCo 版本，可选模型检查缺少资源时跳过，不能把跳过项算作通过。普通遥操作不保存整场诊断，需要时加 `--diagnostic-log`；可选对照入口见[模型说明](docs/MODEL_CALIBRATION_ZH.md)。

[采集手册](docs/COLLECTION_ZH.md) · [ViSk / FTF](docs/LEARNING_WORKFLOWS_ZH.md) · [模型与参数](docs/MODEL_CALIBRATION_ZH.md) · [论文与算法](docs/PAPERS_ZH.md) · [TODO](TODO.md) · [第三方来源与发布事项](THIRD_PARTY.md)
