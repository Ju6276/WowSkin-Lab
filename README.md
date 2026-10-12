# WowSkin-Lab

Franka 触觉遥操作、RGB-D 数据采集和 ViSk / FTF 算法接入。基于 MuJoCo，支持 Panda / Robotiq 2F-85、键盘 / SpaceMouse / PICO 4，提供方块搬运、线材插扣、喂食、Peg Insert，以及实验性薯片夹取场景。

建议从 Panda 方块任务开始。ViSk 提供数据转换、训练与仿真闭环入口；FTF 提供数据导出和标注入口，后续视觉跟踪需要 CUDA。软指腹与磁场是未完成真机标定的本地近似模型，本项目包含数据与环境适配。薯片夹持仍待调整，保留为实验场景；已知问题统一见 [TODO](TODO.md)。

[启动与录制](#启动与录制) · [其他任务](#其他任务与输入) · [数据与学习](#数据与学习) · [代码与检查](#代码入口与维护)

## 安装

当前安装入口面向 Apple Silicon macOS、Python 3.12、MuJoCo 3.15，从源码目录运行。Linux / CUDA 全流程支持仍在 TODO 中。

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
# 无外设：键盘
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

DIGIT、GelSight 是硬件；TACTO、FOTS 是触觉生成算法。**保留 Panda 原手指，将完整传感器模块附加到指尖末端**，感应面朝夹持区；不替换原手指、不添加闭合空隙或夹持附着。

- `tacto`：DIGIT 独立 CAD＋原 TACTO 相机、灯光、背景校正、噪声、力相关光学位移、RGB／深度。
- `digit_fots`：同一 DIGIT CAD＋FOTS 的 **DIGIT 专用**权重、背景和标记网格；原平滑、阴影、膨胀／剪切／扭转模型均执行。
- `digit_compare`：**同一个 MuJoCo 状态同时运行以上两种算法**，并排显示和保存。两者保留各自的投影与深度生成方式，不做增益对齐；区别见[模型说明](docs/MODEL_CALIBRATION_ZH.md#digit-双算法比较)。
- `gelsight`：移植 [GelSightMujoco](https://github.com/rlamprell/GelSightMujoco) 的完整外壳／玻璃／胶面／接触层／内部相机，执行原 Phong 光学、平滑、阴影及纹理噪声。不是此前的矩形代理，也不冒称 GelSight Mini。
- `fots`：保留的旧 **GelSight 标定＋平面代理**选项，便于已有命令兼容；不是 `digit_fots` 或 `gelsight` 的别名。
- `anyskin`：默认磁场传感器，原采集与策略流程不变。

```bash
# 可选安装到同一个 .venv；上游源码／权重／CAD 单独下载
.venv/bin/python scripts/setup_upstream.py --only tacto fots gelsight_mujoco
.venv/bin/python -m pip install -r requirements/tacto.txt
.venv/bin/python -m pip install --no-deps -e third_party/mujoco-tacto

# DIGIT 双算法独立台：可选 sphere / box / cylinder 探头
.venv/bin/mjpython -m wowskin_lab sensor --sensor digit_compare --probe-shape box --ideal
# Panda 双指实时比较；也可 --input spacemouse
.venv/bin/mjpython -m wowskin_lab teleop --sensor digit_compare --task pick_cube --input keyboard --ideal
# 单独算法：将 digit_compare 改为 tacto 或 digit_fots

# 真正独立的 GelSight MuJoCo 模型
.venv/bin/mjpython -m wowskin_lab sensor --sensor gelsight --ideal
.venv/bin/mjpython -m wowskin_lab teleop --sensor gelsight --task pick_cube --input keyboard --ideal

# 同轨迹／同物理状态的数据比较：三种探头，按压、滑动、扭转、倾斜、释放与再次接触
.venv/bin/mjpython -m wowskin_lab sensor --sensor digit_compare --ideal --compare-data --output outputs/diagnostics/digit_compare
# 方块抓取／抬升／释放，保存两指的全部光学输出
.venv/bin/mjpython -m wowskin_lab teleop --sensor digit_compare --task pick_cube --ideal --demo --output outputs/diagnostics/digit_cube

# 独立执行原作者函数的数值对照及接触回归
.venv/bin/python -m unittest tests.physics.test_optical_tactile tests.physics.test_fots tests.physics.test_optical_backends
```

独立台：**Q/E** 下压／抬起，**W/S、A/D** 平移，**I/K、L/J、U/O** 倾斜／扭转，**R** 复位，**P** 退出。Panda 沿用原键盘操作，**X/Z** 闭合／打开。右侧 RGB／深度／FOTS 标记流保持原图宽高比；**M** 切换深度／接触区。深度灰度范围自动共享并标注 µm；`--depth-display-max-mm 5` 可固定范围。只改变显示，不改原始数值。

DIGIT 每指输出 RGB `320×240×3`、深度和有效区 `320×240`；DIGIT FOTS 另有 `8×9×2` 标记像素位移和流图。GelSight 原图是 `120×160×3`，同时保存米制相机深度与原算法使用的 **OpenGL 缓冲值**，两者单位不同。诊断 `--demo` 保存 NPZ／PNG；`--compare-data` 每种探头保存 80 帧 NPZ、逐帧 CSV、MP4 与报告。详见[字段与比较方法](docs/MODEL_CALIBRATION_ZH.md#digit-双算法比较)。

`--ideal` 关闭 TACTO 的随机 RGB 噪声或 GelSight 的纹理噪声；真实背景本身的纹理仍保留。FOTS 不额外加入随机噪声。`--no-fots-shadow` 可关闭原平面阴影；TACTO 支持 `--tacto-config`、`--tacto-background` 和 `--no-tacto-background`。不人为放大输出，不用目标物体白名单过滤触觉。

**本次 sensor 功能范围：独立台、Panda 方块、原算法光学功能和原始输出比较。** 不是只绘制示意图；源码对照测试检查原始 RGB、深度或标记值。GelSight 内部相机固定到来源的深度裁剪参数，避免外部场景尺寸改变光学响应；旧 `mujoco_py` 与当前引擎的整条链路等价性仍待验证。光学图像尚未接入 LeRobot／ViSk，也未接其他任务、Robotiq 或 Pico；这些是单独的应用工作。原算法的刚性接触／光学形变近似不等于材料 FEM；MuJoCo 支持软体，但本次没有用未标定的胶体力学冒充作者模型。安装件和硬件标定状态见[模型说明](docs/MODEL_CALIBRATION_ZH.md#传感器硬件与光学算法)及 [TODO](TODO.md)。

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
- **100 Hz raw**：TCP 基座系 XYZ＋rotvec、线／角速度、七关节位置／速度、夹爪反馈／目标、15 路触觉、基线及质量字段。不是 RPY。
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
tests/        unit、physics、official、artifacts、manual
docs/         采集、学习、物理模型说明
data/         本机示范与算法导出，不进 Git
outputs/      本机权重、标定与诊断输出，不进 Git
```

默认采用 MuJoCo 3.15 降阶指腹＋拟合偶极子。`--ideal` 关闭合成触觉噪声、滤波和延迟，不关闭接触力学。控制限制、参数来源与物理失效规则见 [模型说明](docs/MODEL_CALIBRATION_ZH.md)。完整网格对照针对 MuJoCo 3.5，与当前默认环境不同。

```bash
# 以下检查包含学习接口，先按学习文档获取固定版 ViSk
.venv/bin/python -m tests unit
.venv/bin/python -m unittest tests.official.test_visk_command_labels tests.official.test_lerobot_recording
.venv/bin/python -m tests all --list
```

`unit` 检查坐标、输入和动作契约；`physics` 运行接触场景；`official` 检查官方数据与学习接口；`artifacts` 依赖可选数据／权重；`manual` 需要人工操作窗口。Robotiq 强夹持的已知失败保留在物理测试中，不跳过。物理／artifact 检查可能需要特定 MuJoCo 版本或可选模型，不能把跳过项算作通过。普通遥操作不保存整场诊断，需要时加 `--diagnostic-log`；可选对照入口见[模型说明](docs/MODEL_CALIBRATION_ZH.md)。

[采集手册](docs/COLLECTION_ZH.md) · [ViSk / FTF](docs/LEARNING_WORKFLOWS_ZH.md) · [模型、参数与论文](docs/MODEL_CALIBRATION_ZH.md) · [TODO](TODO.md) · [第三方来源与发布事项](THIRD_PARTY.md)
