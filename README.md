# WowSkin-Lab

Franka + 触觉指腹的 MuJoCo 遥操作与数据采集。支持 Panda / Robotiq 2F-85、键盘 / SpaceMouse / PICO 4；所有任务录制两路固定相机＋一路腕部相机的 RGB-D、机器人状态和 15 路触觉。

**当前建议：先用 Panda cube 继续采集。** 触觉是未做真机标定的仿真信号；Robotiq 强夹持和线材压扣仍可能使软体网格失效。真实 Franka 后端未接入。

## 1. 启动 cube

在本机终端执行：

```bash
cd /Users/judong/Desktop/WowSkin-Lab
.venv-reduced/bin/mjpython -m wowskin_lab teleop --task pick_cube --input spacemouse --ideal
```

这条命令继续追加到 `data/episodes/pick_cube/`，保留已有两条示范。启动后先测试操作，**不会自动录制**。一次只开一个遥操作窗口。

- **SpaceMouse**：点击仿真窗口、旋钮回中；推拉控制 XYZ，倾斜／扭转控制旋转。左侧键闭合、右侧键张开、两键同时按保持。此模式独占 HID，避免旋钮带动普通鼠标。
- **键盘**：把命令中的 `--input spacemouse` 改为 `--input keyboard`，并加 `--record-dir data/episodes_keyboard`，避免混入已有 SpaceMouse 采集。W/S 前后、A/D 左右、E/Q 上下；I/K、L/J、U/O 旋转；X 闭合、Z 张开、C 保持。支持长按。
- **Pico**：见下方 USB 启动步骤。

## 2. 录制与保存

1. 按 **B**：复位并准备录制；看到红色 `REC capture_…` 再开始操作。
2. 将方块从黄框抓起，放到绿框并松开。左上角绿色 **SUCCESS** 是自动任务判定。
3. 按 **Enter**：结束并标记成功；显示 `SAVING → VERIFYING → SAVED`。自动达标不会自行结束录制。
4. 等到 **SAVED** 后，按 **B** 开始下一段；编号自动递增，重启也可追加。

**N** 保存失败；录制中再按 **B** 结束为待审核；**R** 中断并复位；**P** 保存中断记录并退出。写入时不接受下一段录制，退出会等待当前写入结束。保存提示分别显示人工标签和自动判定，`SAVED` 不等于任务成功。

**V** 切换主视角，**H** 展开诊断，逗号／句号调触觉显示比例。半透明成功区域只出现在操作窗口，不进入录制图像。

## 3. 其他场景与夹爪

```bash
# 原绕线布局
.venv-reduced/bin/mjpython -m wowskin_lab teleop --task cable --input spacemouse --ideal
# 简单插线：固定一端、一个反向 clip
.venv-reduced/bin/mjpython -m wowskin_lab teleop --task cable --cable-layout single_clip --input spacemouse --ideal
# Panda 喂食：初始化已夹勺、装好六颗豆子
.venv-reduced/bin/mjpython -m wowskin_lab teleop --task feeding --input spacemouse --ideal
# Robotiq 喂食：使用独立采集目录
.venv-reduced/bin/mjpython -m wowskin_lab teleop --task feeding --gripper robotiq_2f85 --input spacemouse --ideal --record-dir data/episodes_robotiq
```

所有任务使用同一套 B / Enter / N 录制操作。更换夹爪、相机、控制或物理配置时使用新的 `--record-dir`；不要混入现有 cube 数据。Robotiq 初始化已完成接触预夹，直接送食即可；全闭合仍可能过压，未用隐藏行程截断规避。

## 4. PICO 4：佩戴／对面模式

USB 连接 Mac、开启头显 USB 调试并授权后：

```bash
.tools/platform-tools/adb devices
.venv-reduced/bin/mjpython -m wowskin_lab teleop --task pick_cube --input pico --ideal --record-dir data/episodes_pico
```

程序通过 ADB 打开头显浏览器。选择**佩戴模式**或**对面模式**，点击“进入遥操作 / Enter VR”，松开右握键一次，再握住移动。

- 佩戴模式：在头显中看画面；对面模式：头显朝向自己，在 Mac 看画面，停止额外传输头显视频。
- 右握键按住跟随、松开停止；右扳机在握键按住时调夹爪。
- 松开握键后，**右 B** 切模式；**右 A** 在手柄水平指向屏幕时重设操作前方。两种模式按同一画面方向约定控制。
- **左 X** 开始／结束录制，**左 Y** 保存成功，按下左摇杆保存失败；录制操作前松开右握键。
- 对面模式仍需先进入 WebXR，头显保持唤醒与手柄追踪；不是脱离头显的手柄模式。

两种模式的方向、切换、重连和三相机录制有自动测试。实际佩戴舒适度、遮挡与追踪仍需设备实测；当前没有连接 Pico，不能把软件测试说成本次硬件验收。

## 5. 数据检查与训练

```text
data/episodes/pick_cube/
  lerobot/       meta/ + data/*.parquet + videos/*.mp4 + depth/*.h5
  raw/
    capture_000000/  metadata.json + measurements.h5
    capture_000001/  ...
```

- **30 Hz**：三路 RGB、同步米制深度、instruction、对应 state/action。
- **100 Hz raw**：基座系 TCP 位姿／速度、七关节位置／速度、夹爪反馈／指令、15 路磁场、基线、时间戳、相机标定与质量字段。成功保存后不另留 raw 图像副本。
- `state` 保存实测值，`action` 保存实际下发目标。夹爪当前为连续闭合比例（0 开、1 闭），另存驱动关节及单位；**尚未增加米制宽度字段**。
- 失败／中断保留 raw 和恢复用暂存媒体，不进入 LeRobot；写入或验证失败留下 `INCOMPLETE`，不会显示保存成功或继续追加。

检查已有数据：

```bash
.venv-reduced/bin/python -m wowskin_lab dataset-check data/episodes/pick_cube/lerobot
```

采集完成后，按 [ViSk / FTF 学习工作流](docs/LEARNING_WORKFLOWS_ZH.md)选择 episodes、转换和训练。ViSk 使用三视角＋单块触觉；FTF 还需要标注／跟踪。这里不自动启动训练，也没有可沿用的新三视角策略权重。

## 6. 安装与代码结构

新电脑使用 Python 3.12；本机已有环境无需重复安装：

```bash
git clone git@github.com:Ju6276/WowSkin-Lab.git
cd WowSkin-Lab
python3.12 scripts/setup_capture.py
.venv-reduced/bin/python scripts/setup_upstream.py --only franka
```

日常只用根目录 `requirements.txt`（自动引用 `requirements/runtime.txt`）；`requirements/learning.txt` 为学习可选依赖，`requirements/annotation.txt` 仅供 notebook 标注。Pico 另需把 Android Platform-Tools 解压为 `.tools/platform-tools/`，或将 `adb` 加入 PATH。

```text
wowskin_lab/   core/ 物理与磁场；envs/ 场景；teleop/ 输入；ui/ 显示
               data/ 采集与转换；learning/ 官方算法接入；baselines/ 触觉模型
configs/       相机、控制、任务判定、传感器与固定上游版本
assets/        原始素材、派生模型、许可证与来源 manifest
scripts/       安装、素材下载、训练、评估；diagnostics/ 可选模型分析
tests/        unit/、physics/、official/、artifacts/；manual/ 手动窗口检查
third_party/   本机下载的固定版本依赖
data/         用户示范与导出（不上传 GitHub）
outputs/       标定、必要磁场模型、诊断与以后训练结果（不上传 GitHub）
docs/          操作与数据字段、学习流程、模型与标定
```

默认 `.venv-reduced` 使用 MuJoCo 3.15 降阶指腹＋拟合偶极子。`--ideal` 关闭合成噪声／滤波／延迟，不关闭接触力学。默认保留已测试的控制限制；参数、来源及关闭后的影响见 [模型说明](docs/MODEL_CALIBRATION_ZH.md)。此次整理不改这些物理参数或数据动作定义。

普通遥操作不额外保存整场诊断录像／数据；需要时显式加 `--diagnostic-log`。发生物理故障时保留最多 200 条末尾样本和错误原因。正式示范始终通过 B 手动录制。

完整网格参考实验仍使用独立的 MuJoCo 3.5 环境（本机 `.venv`）；不能把其测试直接放进 3.15 环境运行。日常采集统一使用 `.venv-reduced`，不依赖旧的 Python 3.9 安装入口。

维护检查：

```bash
.venv-reduced/bin/python -m tests unit
.venv-reduced/bin/python -m unittest tests.official.test_lerobot_recording tests.physics.test_pico_input
.venv-reduced/bin/python -m tests all --list
```

[详细操作与字段](docs/PICK_CUBE_ZH.md) · [ViSk / FTF](docs/LEARNING_WORKFLOWS_ZH.md) · [参数来源、限制与论文](docs/MODEL_CALIBRATION_ZH.md)
