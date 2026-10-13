# 遥操作与数据采集

所有启动命令见 [README](../README.md)。在项目根目录运行，每次只启动一个遥操作窗口。图形入口使用 `.venv/bin/mjpython`。

## 场景选择

- `--task pick_cube`：黄框抓起、绿框放下。追加 `--randomize` 可在复位时将方块 XY 各随机偏移 ±25 mm，`--seed` 控制序列。
- `--task cable`：本地两柱绕线布局，两个导向柱、右侧黄色卡槽及固定座。
- `--task cable --cable-layout single_clip`：简单插线，保留线材固定座和一个黄色 C 型卡槽；卡槽旋转 180°。卡槽位于初始线束的 +Y 一侧，body 位置 `(0.45,-0.06,0.1009) m`，入口距初始线束约 5 cm；线材首段仍固定。数据任务名 `cable_insert`。
- `--task peg_insert`：SERL++ 原始 peg／孔座素材，启动时已物理夹住 peg。Panda 从 63.6 mm 宽面夹持，Robotiq 因贴指腹后开口较小，从 24.46 mm 窄面夹持；夹持点在源 peg 坐标 z=-35 mm，靠近顶端。松开可自然掉落，无固定约束。
- `--task feeding`：固定机器人送食，勺子已夹住并装有六颗豆子。默认 `--feeding-setting care_room`：人物坐在桌边护理椅上，EBiM 人头连接本地躯干/四肢；暖色木地板、窗帘、柜子和绿植构成康养室内背景。人体/椅子为固定刚性碰撞模型，背景装饰不参与接触；不是人体软组织仿真。原来的独立桌面人头布局可用 `--feeding-setting ebim_station` 打开。
- `--task chip_tray`：Panda、康养房间、蓝绿色托盘与马鞍薯片；夹持仍待调整，操作和范围见[薯片试验](#薯片试验)。

喂食继续显示原有头部接触载荷及峰值，不把躯干碰撞计入头部峰值。人头保持闭嘴，没有口腔或吞咽模型。EBiM 原始文件保持不变；坐姿版只在派生视觉网格中收拢颈部，头部碰撞仍用原始刚性网格。桌面碰撞尺寸/摩擦、勺子和六颗豆子的物理参数沿用原场景。康养布局的人头位置为 `(-1.09, 0.04, 1.36) m`，朝向机器人；布局、人体模型范围、头部位置和三路相机实际内外参进入采集元数据。新旧布局的画面分布不同，不视为同一视觉实验条件。自由视角转到墙外时，显示暂时隐藏挡视线的墙面/窗帘；固定采集相机始终渲染完整房间，碰撞不随视角变化。

**界面与保存：**五点触觉在左下角，右侧三路预览使用实际采集相机。有自动评分的任务达到条件后，左上角持续显示 `SUCCESS`，直到复位；薯片没有自动成功判定。B 手动开始录制，Enter 手动结束并标记成功；自动判定不会启动、截断录制或控制机器人。人工标签与每帧自动评分分别保存，允许复核二者是否一致。

**方块和线材成功判定：**参数在 [configs/task_success.json](../configs/task_success.json)，是本地任务定义，不是论文或 EBiM 官方阈值：

- 方块初始完整投影在黄框内，被夹持抬高 ≥60 mm；随后八个角点的 XY 全部进入绿框内沿、底面距桌面 ≤2 mm、脱离所有夹爪接触、线速度 ≤15 mm/s 且角速度 ≤0.15 rad/s，连续保持 0.3 秒。框的位置与视觉标记使用同一份配置。单纯推入绿框、仍夹着或悬在框上方都不算成功。
- **单卡扣 `single_clip`**：线芯中心线必须跨越槽两端及中间截面，且在完整内部通道内连续保持 0.3 秒。v2 在 clip 局部坐标的 X/Y/Z 范围为 [1.5,11.5] / [8.5,20.0] / [1.0,20.7] mm；Y/Z 允许 0.3 mm 接触容差。透明框与评分共用通道边界，不吸附或移动线材。无需松夹爪，这项评分不证明松手后保持。
- **双柱长线 `routing`**：按线段从固定座 0 到 1 的顺序，经过第一柱 +X 侧、第二柱 −X 侧的六个引导截面，在柱帽下方保持 0.3 秒。范围见 [cable_routing.json](../configs/cable_routing.json)。卡扣入槽也会记录，但不决定此任务的 `SUCCESS`；不额外要求松手或插扣。

默认卡扣是刚性开口碰撞代理，可选上下唇弹性梁仍属未标定实验。这些都是本地几何评分，不能作为真实弹性扣合的成功率。

**送食成功判定：**参数集中在 [configs/feeding.json](../configs/feeding.json)。按用户要求，只检查三项：**勺子前部与嘴边区域相交、六颗豆子仍在勺内、连续保持 0.5 秒**。黄色透明框包住勺子前端 20 mm，用这个随勺运动的有向包围盒与嘴边椭球相交来评分，不再限定一个勺尖点；这是评分几何近似，不是新增碰撞。嘴边椭球半径 18/12/12 mm，中心距唇面 12 mm。嘴位置取源人头网格在 x=0、z=-0.4 mm 的前表面，没有改碰撞来让评分通过。初始夹持点沿勺柄向勺碗移动 5 mm，只改变初始化摆放，随后由正常接触求解承重。

勺面倾角、进给朝向、速度、双指夹持状态和头部接触继续记录，**不阻止 SUCCESS**。因此“完成送达”和“是否撞到头部”可以分别比较。头部接触次数、瞬时力、采样区间峰值、全程峰值、合力和接触冲量随 100 Hz raw 的 `evaluation/` 保存；它们是当前刚性头部碰撞代理的仿真量，不能作为口腔软组织力或吞咽判定。配置里的 20°/25°/20 mm/s 保留为诊断参考，不再是完成门槛。

界面 `To finish` 只显示进框、豆子与停留相关的未满足条件。绿色 SUCCESS 保留到复位，不会停止机器人或自动结束录制；Enter 保存人工成功标签，自动评分独立保存。新配置版本是 `local_feeding_delivery_v4_spoon_front`；旧数据保留原判定配置与标签，按旧元数据复现时仍使用旧条件。`delivery_distance_m` 是前部包围盒上最接近目标椭球中心的点到中心的距离，不再是固定勺尖距离。

手动采集默认没有运行限时。`--seconds` 仅用于限时检查，到时会把仍在进行的录制保存为 `interrupted`，不会自动标成功。B 开始，达到绿色 SUCCESS 后 Enter 完成保存；这时会显示 `Saved SUCCESS`。退出前未完成的原始状态、视频、深度仍保留在 raw 和 `.pending_media`，不会丢弃或自动当作成功训练样本。

线材和喂食使用本地布局及 [EBiM 原素材](../assets/ebim/manifest.json) / [喂食素材](../assets/ebim_feeding/manifest.json)，不代表完整官方评测任务。卡槽外观网格和线材材料参数保留来源配置，v2 的三个碰撞块按实际可见通道对齐；短线长度为 0.38 m。本地两柱绕线布局和单卡扣布局共用此碰撞／评分修正。

当前线材沿用 EBiM 的 `mujoco.elasticity.cable` 插件：半径 3.5 mm、密度 800 kg/m³、`bend=2e6`、`twist=1e7`、关节 damping=0.15。原文件是 3 m、`count=71`。`single_clip` 为 0.38 m、24 段，首段固定、末端自由；`routing` 为约 0.756 m、48 段，连接两个固定座。机器人、软指腹及整个场景的积分/接触设置与上游不同，因此不是原场景的等价缩放。独立 `whip` 对照来自另一篇鞭子工作，没有替换此线材。

上游默认 XML 的板面约 1.2×0.93 m，放置四个绕线柱、一个 C-clip 和两个连接座，由移动双臂 FR3＋Robotiq 操作。本地 `routing` 保留两个柱、一个卡扣和两个连接座。首段沿用固定安装，另一端用 MuJoCo `connect` 将线端位置连接到第二座，保留转动自由度；这是两端安装关系，不是夹爪抓取辅助。初始 S 形由顶点定义，插件 `flat=true` 表示无应力形状仍为直线，不会把 S 形当成弹簧的自然形状。位置、离散数、路线判定都记录在 [cable_routing.json](../configs/cable_routing.json)。旧 `ebim_task1_single_arm` 配置仍可重建原单端短线。

导向柱用于限定走向、提供转弯支撑和控制弯曲路径，通常不负责夹紧；它不是额外的连接点。本地两柱目标按从连接座 0 到 1 的线段顺序：经过第一柱 +X 侧、第二柱 −X 侧，每柱三个 Y 截面，并保持在线帽下方 0.3 秒。界面显示六个对应引导截面；不要求额外插扣或松手，卡扣指标单独记录。这是本地几何路线评分，未测量真实布线质量或复现 EBiM 完整评分。初始长线、双端连接和机器人接触需要逐步换抓，不能当成自由末端插线任务。


**Peg 成功判定：**末端双插脚的中心参考点进入孔口至少 20 mm、不超过孔座 50 mm 高度，XY 各偏差不超过 2 mm，peg 与孔座姿态差不超过 5°，保持 0.3 秒。UI 同时显示深度、偏差、角度与保持时间，透明框仅显示位置范围，不进入录制相机。这些是本地几何评分参数，不是 SERL++ 原版 TCP 位姿奖励；无需松开。原始 peg 经 MuJoCo 编译质量约 255 g。机械臂默认用本地 IK＋关节 PD；`--arm-controller cartesian_impedance` 可选固定版 SERL++ 笛卡尔阻抗，reset 仍用原初始化流程。

## 控制与视角

先点击窗口。三种输入默认参照固定机器人基座，转动预览不改变方向。W/S 或上下箭头沿基座 ±X；A/D 或左右箭头沿基座 −Y/+Y；E/Q 沿基座 ±Z。I/K、L/J、U/O 分别绕末端局部 X/Y/Z 轴旋转。支持长按，松开停止。本地工作空间盒子默认关闭；显式 `--workspace-limits` 才开启，H 显示范围。速度和领先量仍生效，见[控制说明](MODEL_CALIBRATION_ZH.md#生效的控制与质量检查)。

普通任务 X/Z 闭合/张开，C 保持当前开度。方块/喂食默认采用 Menagerie Panda 位置伺服；执行器的 ±100 N 是上限而非持续夹持力。线材采用 [robosuite Panda](../assets/reference/robosuite_panda/manifest.json) 的每指独立位置伺服：1000 N/m、每指 ±20 N，并保留其关节阻尼、惯量和摩擦参数；本地安装位置及行程另行保留。这是基准控制参数，不是真机接触力标定。`--grip-force` 才显式启用旧力限制器；普通任务默认的 -/= 不调力。薯片是单独的力反馈实验，按键语义见下节。

V 切换近景/全景，Esc 恢复全景；鼠标左拖旋转、右拖平移、滚轮缩放。逗号/句号调整触觉显示比例。R/Backspace 复位，P 保存并退出。F 仅用于方块自动短测，不能用于线材或喂食。

左侧默认只保留输入状态、任务进度、录制状态和简短操作提示。H 展开/收起磁场模型、夹爪驱动、接触力、网格质量、求解耗时及 IK 等诊断；收起只影响文字显示，不影响物理计算、录制字段或三路相机/触觉。录制异常、网格警告和物理故障仍会提示。

若显示 `PAUSED: physics invalid`，物理求解已停止，触觉面板保留最后有效帧；按 R 复位后继续。该段录制会中止，不导出失效后的物理数据。

## 薯片试验

该场景仍待调整，暂未验证稳定无损夹取，不作为已完成的训练任务。启动命令见 [README](../README.md#脆性薯片试验)。机器人版约 70×50 mm，马鞍高低边缘相差 24 mm，由 24 个刚体薄片、28 条可断连接组成；尺寸、强度及断裂模型范围见[模型说明](MODEL_CALIBRATION_ZH.md#脆性薯片预分块试验)。

- X／SpaceMouse 左键启用夹爪接触力闭环，Z／右键打开；C／双键继续保持力闭环。
- `-` 降低、`=` 提高目标力，每次 0.01 N，不按 Shift；Pico 扳机映射为 0～所选目标力，松开打开。
- 默认目标 0.05 N；可显式加 `--chip-grip-force 0.08` 调整。默认空载时每指目标速度约 5 mm/s，上限 10 mm/s，接近力目标时减速。
- V 切近景，R 复位；B/Enter/N 仍可手动录制。界面显示 `INTACT / BROKEN` 与碎片组数，没有自动 SUCCESS，Enter 仅表示人工标签。

此闭环只接入薯片的 Panda 夹爪，机械臂默认仍是 IK＋关节 PD。反馈取 MuJoCo 两指全部接触的法向载荷中较大者，不是由 15 路磁场估计的力；两指互碰也可能达到目标。默认力不保证不滑落或不破裂。ViSk 仍保持触觉输入、末端运动＋夹爪开合输出，未添加目标力动作。

raw 额外保存 `gripper_force_feedback/` 中的力目标、启用状态、双指载荷、`width_m` 实测宽度和 `servo_target_width_m` 伺服目标宽度。宽度来自 Panda 驱动关节，不是变形后接触面间距；公共 state/action 仍使用实测／指令闭合比例。

```bash
# 独立按压台，无机器人，不自动录制
.venv/bin/mjpython -m wowskin_lab brittle-chip
# 自动按压、抬起及断裂事件，仅保存到本机 outputs
.venv/bin/python -m wowskin_lab brittle-chip --headless --video --output outputs/diagnostics/brittle_chip
# 机器人夹持诊断；结果不作为策略成功率
.venv/bin/python -m scripts.diagnostics.check_chip_force --forces .05 .08
```

独立按压台的 Space 执行／停止按压目标，D/U 下压／抬起，R 重置，P 退出；默认 24 块／32 条连接，与机器人版分块不同。剩余工作集中在 [TODO](../TODO.md#薯片夹持待调整暂不作为训练主线)。

## SpaceMouse USB 遥操作

使用 `--input spacemouse`。旋钮前后映射为基座 ±X、向左映射为基座 −Y、向右映射为基座 +Y、提起为基座 +Z，倾斜/扭转控制三维旋转。普通任务左键夹紧、右键张开、双键保持开度；薯片见上节。旋钮回中停止，失去窗口焦点停止输入。

macOS 下此模式会独占打开 SpaceMouse，终端应显示 `exclusive HID=True`。若误用 `--input keyboard`，程序不会读取/独占 SpaceMouse，旋钮可能仍移动系统鼠标；请退出并按 README 的 SpaceMouse 命令启动，不需要关闭普通鼠标。

前后倾斜绕固定基座横轴旋转；不再随观察相机改变。左右平移按面对机器臂的操作约定（左 −Y／右 +Y）；这是固定输入映射，不修改机器人或数据的基座坐标系。输入轴符号保留此前确定的前压／后仰约定，并保存在 metadata。显式 `--control-frame view` 才使用旧视角方向。键盘旋转键绕 TCP 局部轴，SpaceMouse 角速度先按基座轴解释再转换为 TCP 局部增量。

默认平移上限为 0.12 m/s、角速度 0.5 rad/s，均按仿真时间计。可设置 `--spacemouse-speed`、`--spacemouse-angular-speed`、`--spacemouse-deadzone`。面板的 `Simulation speed` 是实际实时倍率；换输入设备不会减少软体求解耗时。

## Pico USB 遥操作

使用 `--input pico`，USB 连接头显并开启 USB 调试。程序建立 ADB 转发并打开头显浏览器；选择佩戴/对面模式，点击“进入遥操作”，先松开一次右握键。

- 右握键：按住相对移动/旋转，松开保持，再握住建立新起点。
- 右扳机：握键按住时调夹爪闭合度。
- 右 B：握键松开时切换佩戴/对面模式。
- 右 A：握键松开、手柄水平指向操作前方时重新校准；默认将该方向映射为机器人基座 +X，手柄左／右平移分别为基座 −Y／+Y，与预览角度无关；左右平移镜像不改变已约定的旋转映射。
- 左 X：开始录制或结束为待审核；左 Y 保存成功；按下左摇杆保存失败。录制按钮操作前松开右握键。

对面模式只在 Mac 显示完整仿真，头显仍需保持唤醒和 WebXR 追踪。追踪中断或重定位后，松开再握住右握键恢复。右手控制单条 Franka，左手用于录制；此记录属于 FTF 机器人分支。

Pico 另需 Android Platform-Tools：将 `adb` 加入 PATH，或解压为 `.tools/platform-tools/`。佩戴模式向头显传输画面，对面模式只显示桌面窗口；二者都需要先进入 WebXR，头显保持唤醒与追踪。软件测试覆盖方向、模式切换和重连，硬件表现需在目标设备上验收。

连接排查：`.tools/platform-tools/adb devices -l` 应显示一台已授权设备。服务重启后重新进入 VR；旧页面连接失败时刷新页面。

## 在线 RGB-D 点云

在线点云预览使用共享遥操作入口：在启动命令后加 `--pointcloud`，点云显示在右侧三路 RGB 下方，与相机同列。也可在窗口按 **M** 切换两路固定相机／三路含腕部／关闭。**Y** 换观察角度，**0** 切换 RGB／相机来源色，**9** 缩放点云面板。融合使用同一仿真时刻的深度和逐帧相机外参，输出到机器人基座系；不跨时间累加、不直接从物体网格采样点，也不进行学习式点云补全。深度来自 MuJoCo 的理想渲染，目前没有模拟真实深度相机的噪声／缺测。

预览默认最多 5 Hz，每隔 4 像素采样、每路做 2 mm 体素降采样，可用 `--pointcloud-hz`、`--pointcloud-stride`、`--pointcloud-voxel` 调整。融合和点图绘制在后台，RGB-D 渲染仍有额外开销；面板显示采图及融合耗时。原始 RGB-D 录制分辨率、时钟、控制参数不变，点云面板不会进入相机图像，也不额外写入数据集。

## 五个点的 XYZ 怎么看

面板显示五个磁力计的三轴差分磁场，单位 μT；绿色线段表示 XY，红色圆圈表示 Z 的符号与幅值（正值实心、负值空心）。默认比例为 7 μT/逻辑像素。它不是三维力。

原偶极子与拟合偶极子使用 U1中心、U2上、U3右、U4下、U5左的 PCB 参考顺序。网络预览使用 CH1中心、CH2左、CH3右、CH4上、CH5下的候选分组。实物对应与轴向仍未全部验证。

各任务遥操作默认使用 `configs/magnetic_fit_pollen_nonslip.json` 拟合偶极子：网格形变计算磁场，再应用五组幅值增益。需要网络对照时追加 `--reskin-model outputs/reskin_forward_mapped`，严格检查输入且不自动外推；其输入是所有外部接触的中心 XY 和最大法向压缩深度，不是完整形变场。`--reskin-extrapolate` 显式开启未经验证的接触预览；无接触仍为不可用，不人为返回零。拟合与网络两者互斥，不自动回退。网络数值诊断保留同状态 `fitted_dipole_ideal_delta_ut`（原 U 顺序），但不混入主观测。详见 [模型与标定](MODEL_CALIBRATION_ZH.md)。

## 保存了哪些数据

所有任务只采集一次，默认 `--record-format lerobot`。`data/episodes/<任务>/lerobot/` 是成功 episodes 的 LeRobot v2.1，`raw/capture_000000/` 等目录只保存 `metadata.json` 与 100 Hz `measurements.h5`；RGB/深度最终只保留在 `lerobot/`，metadata 用相对路径引用；简单插线任务名为 `cable_insert`。失败/中断只留 raw，不进入 LeRobot 训练数据。`--record-dir` 指定采集根目录；显式 `--record-format dataset` 只保存原始格式。

LeRobot 使用未修改的 `lerobot==0.3.3` 写入器/读取器（v2.1），数值存 Parquet，图像为 MP4。`observation.state` 为实际基座系 TCP xyz＋rotvec＋夹爪，`action` 为该观测之后第一条 100 Hz 控制发出的绝对 TCP/夹爪目标，另存 `observation.joint_position`、`observation.joint_velocity`、`observation.tcp_velocity` 和 15 路 `observation.tactile`。这是本项目的动作契约，不自动等同于任何 VLA 的预训练动作空间。

每行对应一帧实际采集图像，无重复补帧。LeRobot `timestamp=frame_index/30` 用于视频索引，`aux.sim_timestamp` 保留真实仿真采样时刻；相机在 100 Hz tick 上触发，两个时间轴最大偏差约 6.67 ms。`aux.raw_sample_index`、`aux.raw_camera_index` 可回查原始值，`aux.action_timestamp` 核对动作时序。100 Hz 触觉、基线和内外参完整保留在 raw；FTF 导出读取该附件。写入异常会留下 `INCOMPLETE` 标记，不能将半成品用于训练。

H5 的机器人状态和触觉按 100 Hz 仿真时间同步保存，三路相机（固定 1、2，腕部 52）按约 30 Hz 保存：

- `cartesian_states` / `commanded_pose`：实际／目标 TCP **基座系** XYZ（米）＋旋转向量（弧度），不是 RPY。基座是 Panda `link0`；TCP 是新指腹之间的 `tactile_tcp`，不是法兰中心。
- `arm_qpos` / `arm_qvel`：七关节实际角（rad）／实际角速度（rad/s），顺序 `joint1` 到 `joint7`。不是目标关节角。
- `tcp_velocity`：实际基座系线速度 vx/vy/vz（m/s）＋角速度 wx/wy/wz（rad/s），由 MuJoCo Jacobian 与实际 qvel 计算；不是 rotvec 的逐项差分。另保留世界系 `tcp_velocity_world`。
- `gripper_states` / `commanded_gripper`：实际／目标闭合比例；另存夹爪驱动关节的实际位置、速度及对应单位。
- `command_7d`：基座系 ΔXYZ＋目标 TCP 局部系 δrotvec＋夹爪闭合目标。控制器内部仍用世界系，`command_7d_world` 保留实际下发的世界系命令。
- `sensor_states`：15 路含基线的仿真磁场（μT）；基线、通道、拟合增益及物理参数写在 metadata。未完成实物标定。
- `world_from_base`、`cartesian_states_world`、`commanded_pose_world` 保留换算依据；`camera_base_to_cv` / `camera_world_to_cv` 是同一相机在两种参考系下的外参，内参单独保存。
- `phase`：0 为实际预录制、1 为演示、2 为实际收尾。采集包括 0.6 秒预录制、至少 0.2 秒收尾，必要时增加一条实际控制采样以保存末帧之后的真实动作；为 FTF 保留实际的触觉历史及暂停边界，不复制帧。异常中止不推进仿真补收尾。
- `skin_min_volume_ratio`、`passive_min_volume_ratio`、`mesh_quality_warning`：与 100 Hz 状态同频的网格质量诊断。0–15% 的正体积区间只警告，继续实时计算磁场和录制，不保持旧值；字段保留在 raw H5，可按相机时间映射回 LeRobot 帧。当前算法导出不因警告自动删除帧。单元翻转或求解发散才中断该段采集。

方块/线材的基座与世界系重合；喂食中基座在 `(-1.63,-0.38,1.08485019) m`、绕 Z 转 90°，采集时使用 `T_base_tcp=inv(T_world_base)·T_world_tcp`。工具相对 hand 的位置与朝向也写入 metadata；与真实 Franka / 官方 FTF 工具参考点的注册仍未验证。

启动时不录制，B 复位并开始；Enter 保存成功并编码 LeRobot 视频，N 保存失败；P 保存中断并退出。编码期间有等待和终端提示。成功标签由操作者给出。原始记录不因算法而重复保存，后续使用 `dataset-export` 选择成功 episodes，生成 ViSk 或 FTF 的派生输入，见[学习工作流](LEARNING_WORKFLOWS_ZH.md)。关节数据保留在公共源数据中；不会擅自把它们加入官方策略网络输入。

原始附件 schema 为 `wowskin_robot_dataset_v1`；它与 LeRobot v2.1 容器版本是不同概念。旧文件仍保留原坐标约定，不原地改写；世界系与基座系记录不能混入同一个导出。完整采集流程测试可用默认拟合偶极子；显式切换到网络后，缺失的卸载观测会使录制被拒绝。

## 光学触觉数据

选择与安装命令见 [README](../README.md#可选视触觉传感器)。任务窗口沿用本手册的 B／Enter 录制流程；**逗号 `,`** 切换深度／接触区，M/Y 保留点云功能。深度自动标注 µm，可用 `--depth-display-max-mm 5` 固定显示范围，保存值不缩放。

独立按压台支持 `--probe-shape sphere|box|cylinder`（这是三种可选测试压头，不是每指三个传感器）。**Q/E** 下压／抬起，**W/S、A/D** 平移，**I/K、L/J、U/O** 倾斜／扭转，**R** 复位，**P** 退出，**M** 切深度／接触区。`--no-fots-shadow` 仅关闭 FOTS 源阴影；TACTO 没有 FOTS 的 marker 算法，界面不生成虚假流图。

```bash
# 真实接触夹起、保持、释放的脚本诊断；切换 --optics 分别运行
.venv/bin/python -m wowskin_lab teleop --sensor digit --optics fots --demo --ideal
# 源机械参数、原函数输出、按压卸载、跨任务录制及官方读回检查
.venv/bin/python -m unittest tests.physics.test_digit_sources tests.physics.test_optical_tactile tests.physics.test_fots tests.physics.test_optical_backends tests.physics.test_optical_tasks tests.official.test_optical_recording
```

`--demo` 保存 NPZ／PNG；`digit-compare` 对球、方体、圆柱分别保存两算法各自状态、指令、RGB、深度、力、标记、CSV 与 MP4 到本地 `outputs/diagnostics/`。两套机械模型独立运行，同指令不等于同接触力；真实性排名需要配对实物数据。

`--sensor digit --optics tacto|fots` 或 `--sensor gelsight` 每次选择一种，两指同型号。左右固定对应机器人 `left_finger`／`right_finger`，不随相机改变。五类 Panda 任务共用上述 B／Enter／R 流程、机器人基座坐标与连续夹爪指令。

光学录制与 AnySkin 磁场协议分开：raw 仍记录 100 Hz 机器人、关节、动作与双指法向接触力 `optical_contact_force_n`；没有 `sensor_states`、磁场基线或 FEM 网格字段。真实光学帧与三路 RGB-D 在相同的相机时刻采样，约 30 Hz，不重复帧伪造 100 Hz。LeRobot schema 为 `wowskin_lerobot_optical_v1`。

- `observation.images.tactile_left/right`：双指 RGB 视频。DIGIT 每指 320×240×3，GelSight 120×160×3；shape 顺序为高、宽、通道。
- `lerobot/optical/chunk-000/episode_000000.h5`：`timestamp` 与 `frames/000000/left|right/`。保存无损原生 RGB、`indentation_m`、原始相机／空载深度、接触／有效掩码、原算法缓冲及其单位、力、传感器相机位姿和源版本。无损 RGB 与视频并存是用于源算法比较的明确保留；raw 不再另存图像。
- FOTS 在来源支持时另存 `marker_displacement_px`（9×11×2）与流图；`marker_status` 明确区分无接触、接触和多 body 不支持。TACTO／GelSight 不伪造 marker，多物体 FOTS 不用零填充冒充有效数据。
- `observation.tactile_contact_force`：双指 MuJoCo 接触法向力诊断，单位 N，不是真机传感器直接读出的力。不自动作为策略输入。
- 100 Hz state/action、关节、速度、相机基座系变换、instruction 与磁场录制沿用同一契约。成功段编码后通过官方 LeRobot 读取和无损光学时间戳校验，随后可开始下一段；中断保留 raw 与暂存媒体。

光学数据不能直接送入现有 ViSk／FTF 15 路磁场转换器；入口会明确拒绝，后续需独立图像编码器适配。`dataset-check` 检查五路视频和光学扩展。切换传感器／算法、装配、光学参数后使用新目录；默认分别为 `data/episodes_tacto`、`data/episodes_digit_fots`、`data/episodes_gelsight`。

## Instruction、RGB-D 与多次采集

- `--instruction "Pick up the cube and place it on the table."` 逐字保存；未指定时使用任务模板，并记录来源。一个 episode 使用一条指令，换指令后可继续追加到同一物理配置的数据集。`meta/tasks.jsonl` 保存文字，Parquet 中的 `task_index` 和 `annotation.human.action.task_description` 保存索引；后者沿用 GR00T 命名，不表示示范一定由人操作。
- [GR00T N1 官方数据约定](https://github.com/NVIDIA/Isaac-GR00T/blob/755876a9afdb41ca6eb6383b36f4a0adb085c73f/getting_started/LeRobot_compatible_data_schema.md) 使用 MP4、Parquet、任务索引及 `meta/modality.json`。本项目写入位置、axis-angle 旋转、夹爪、关节、触觉的字段映射；这不代表 GR00T 策略已经训练/部署，也没有改动原版 ViSk/FTF 的输入选择。
- 默认三路 RGB/深度（固定相机 1、2 和腕部相机 52）为同一 MuJoCo 状态、同一相机、640×480 像素。深度在 `lerobot/depth/chunk-000/episode_000000.h5`，float32 **米**，为相机光轴 +Z 深度，不是射线距离；LZF 无损压缩，保存有效掩码、内参、基座/世界外参、实际时间戳。它是 MuJoCo 真值，不是真实深度相机噪声仿真。
- 深度是 v2.1 数据集的扩展，原版 LeRobot/GR00T N1 不会自动加载。通过 `wowskin_lab.data.depth.load_depth_frame(root, episode_index, aux.raw_camera_index)` 读取。不能用训练行的 `frame_index` 猜测原始深度索引。
- LeRobot 保留真实预录制/演示/收尾的全部 RGB 帧；`aux.acquisition_phase` 分别为 0/1/2，ViSk 导出只选演示阶段。这样 FTF 可直接重用同一视频，raw 不必另存 AVI。视频索引固定 30 Hz，另存真实采样时刻，无图像/动作补帧。
- 原始编号 `capture_000000` 按所有采集递增；LeRobot `episode_000000` 只按成功采集递增。失败会使两者编号不同，`metadata.json` 的 `lerobot_episode_index` 与 `meta/wowskin_sources.jsonl` 明确映射。重启会扫描已有记录并追加，重复导入拒绝，同目录并发写入有文件锁。
- 成功编码并校验帧数后移除暂存图像；raw 仅剩数值 H5/metadata。失败/断电的 `.pending_media/` 保留恢复资料，不进入训练集；`INCOMPLETE` 表示打包未完成，需另建导出目录恢复，不能继续向半成品追加。
- 改变物理/相机/传感器配置、深度开关或数据字段版本时需换 `--record-dir`，不能静默混成同一组归一化数据。旧目录不原地升级或补造速度/深度字段。

## 自动采集

```bash
.venv/bin/python -m wowskin_lab record-auto --output data/automated/my_run --episodes 2 --randomize --magnetic-fit configs/magnetic_fit_pollen_nonslip.json --ideal
```

使用新目录，并显式选择观察模型；不再自动选择网络并开启外推。网络缺失空载响应时，此入口会在创建数据目录前报错。该命令是方块脚本示教，不能用作线材/喂食自动成功验证，也不代表磁场已标定。

## 诊断与边界

默认 AnySkin 遥操作使用 MuJoCo 3.15 降阶指腹、CG 与拟合偶极子；显式光学选项使用各自来源的刚性接触／光学算法和求解设置。材料、触觉和夹持仍未做真机标定。完整参数来源及开关影响统一见 [模型与标定](MODEL_CALIBRATION_ZH.md)，不要把脚本测试通过理解为任意接触稳定。

正常遥操作只在按 B 后保存正式采集。完整 NPZ/CSV 诊断需显式 `--diagnostic-log`；故障自动保留最后最多 200 条样本，位于 `outputs/diagnostics/<task>/teleop_…/`。

三路相机布局可用 `.venv/bin/python scripts/preview_cameras.py --help` 查看；原两路与四路参考布局保存在 `configs/cameras.json`，不会自动改动当前采集配置。切换配置必须用新数据目录。
