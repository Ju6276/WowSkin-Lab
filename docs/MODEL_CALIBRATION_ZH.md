# 传感器模型与标定

本项目的 WowSkin 是本地物理近似模型，尚未完成实物材料与磁场标定。启动与录制见 [README](../README.md) 和 [操作说明](COLLECTION_ZH.md)。

## 计算方式来自哪里

AnySkin 的磁化皮肤受接触后变形，改变下方五个三轴磁力计的读数。[AnySkin 论文](https://arxiv.org/abs/2409.08276)

本项目用 MuJoCo 弹性四面体网格近似皮肤形变，再选择磁场输出方式；它不是 AnySkin、ViSk 或 FTF 官方提供的传感器仿真。三个方式共享同一套接触力学：

- **原偶极子（底层参考）**：每个四面体中心放置磁偶极子，位置随网格移动，方向随形变梯度的旋转部分改变；叠加五芯片处磁场并扣除基线。遥操作入口不默认选择此项。
- **拟合偶极子（三个任务的遥操作默认）**：使用 `configs/magnetic_fit_pollen_nonslip.json`，在偶极子场上应用五个正幅值系数。运行时不查询网络；也可用 `--magnetic-fit` 显式指定配置。
- **网络（显式对照，严格模式）**：追加 `--reskin-model outputs/reskin_forward_mapped`。从所有外部接触提取力加权 XY 和最大顶部压缩深度，输入本地 `3→64→64→15` 网络；没有输入完整网格、剪切、接触面积或物体形状。接触不限于任务物体。默认不自动外推；显式加 `--reskin-extrapolate` 仅启用实验接触预览。

拟合与网络选项互斥。`--ideal` 关闭噪声、漂移、滤波和延迟，不关闭接触形变。非 ideal 偶极子路径保留配置中的滤波与延迟；网络没有相应动态模型。

现有网络没有空载与卸载恢复模型；无查询时返回 NaN，并显示不可用，不是预测零。严格模式对未验证接触也返回 NaN。训练录制拒绝这些帧并中止该段，不切换后端、不插补。有限的外推预览仍标为未验证。要得到通用的“完整形变→15 路磁场”模型，现有三输入权重和单压头数据不足，不能仅修改接口来宣称已实现。

### 磁场公式

体元初始磁矩为 `m_j = M V_j e_z`，随材料旋转但大小固定。每个芯片叠加：

`B_i = μ0/(4π) Σ_j [3 r̂_ij (m_j·r̂_ij) − m_j] / |r_ij|³`

芯片位置减去体元中心得到 `r_ij`，长度用米；T 转成 μT 后输出基线差值 `ΔB`。实现：[physics.py](../wowskin_lab/core/physics.py)。均匀磁化、点偶极子离散和材料参数均为模型选择，未验证真实磁滞或网格收敛。

## 当前仿真幅值拟合

运行配置：[magnetic_fit_pollen_nonslip.json](../configs/magnetic_fit_pollen_nonslip.json)。按 PCB **U1中心/U2上/U3右/U4下/U5左** 顺序，倍率为 **54.525 / 15.870 / 50.256 / 65.217 / 27.875**。

拟合参考是 `Pollen 不滑移磁场 → 本地 MagNet 架构逆模型的 XYZ 伪标签 → 已训练前向网络 → 五芯片模长`。数据来自 cable.csv 与仿真手动夹持记录；Pollen 标签 0 不保证夹紧，因此使用最大芯片模长 ≥50 μT 的筛选规则。Pollen 不提供位置、深度或力真值。

两组数据并非同一接触实验，因此拟合的是各芯片模长的 10/25/50/75/90 百分位。每个芯片三个轴乘同一正数，保留方向、符号和零点；基线一起变换，噪声在缩放后单独添加。五个读出倍率不代表已辨识的材料磁化强度，且不会增大夹爪力。

拟合按完整片段／时间分割训练、验证、测试，目标是幅值分布，不是逐帧 15 路磁场。此配置仅在线材上拟合，勺子／方块的迁移与真机精度未验证。

现有倍率来自旧网格，降阶轮廓加密到 48 点后尚未重新拟合；网格细化会改变磁场积分和接触离散，需要重新检查幅值一致性。

[拟合程序](../scripts/diagnostics/fit_pollen_dipole_amplitude.py)保留用于重跑对照；必须显式提供 `--episode PATH/episode.npz` 和新的 `--output`。详细输入、划分与原始报告摘要保存在配置中；完整实验输出不随仓库分发，需要先准备 Pollen 数据与可选网络。默认遥操作只读取已提交的增益配置，不依赖这些权重。

## 参数来源

- 外形：OnRobot 参考 STL 按厂商平面图缩放至约 42.2×43.6 mm；底座约 11.5 mm、套硅胶后总高约 13.5 mm 来自用户测量。顶部增厚取差值 2 mm。用户确认只包上沿，具体侧壁厚度和下包长度尚未测量。
- 材料：杨氏模量 30 kPa、泊松比 0.35、密度 1000 kg/m³、摩擦系数 0.6；弹性阻尼 1e-5。完整节点模型使用局部节点阻尼 0.002，降阶任务将节点关节阻尼设为 0。这些是未辨识的本地初值，不是厂商参数或经过验证的经验值。
- 磁场：均匀 +Z 磁化 4000 A/m、芯片至软层底面间距 2.2 mm，均未实测辨识。
- 信号：100 Hz；独立高斯噪声标准差 0.5 μT、一阶滤波时间常数 10 ms、延迟 1 个采样。噪声不累加，线性漂移默认关闭。
- 标签：接触阈值 0.005 N、滑移速度阈值 0.0005 m/s，为本地标签规则，不是 Pollen 的人工标注。
- 数值：默认 trilinear 降阶使用 MuJoCo 3.15、discrete、CG、500 μs，不启用内部接触。完整网格对照使用 MuJoCo 3.5；其线材/喂食启用内部接触。线材接触半径 0.2 mm、时间常数 0.2 ms、`solimp=0.98 0.98 0.001 0.5 2` 都是本地数值设置，不是实测材料。尚无完整收敛验证。[MuJoCo 内部接触说明](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html#flex-contact)

配置：[sensor.json](../configs/sensor.json)、[sensor_hardware.json](../configs/sensor_hardware.json)。

## 夹爪配置来源

[gripper_control.json](../configs/gripper_control.json) 使用固定版本 Google DeepMind Menagerie Panda 位置执行器，来源与哈希在 [manifest](../assets/reference/panda_menagerie/manifest.json)。原位置增益 100 N/m、速度增益 10 N·s/m、tendon 限幅 ±100 N，两指传动各乘 0.5。

本地关节坐标 `q_local=q_source−0.04`，闭合指令方向相反；实际 tendon 输出为 `−4×closure−100×q_local−10×qdot`。因此 ±100 N 不是恒定夹紧力。新增软指腹的接触负载仍可能顶开位置伺服；不能用磁场倍率解决该问题。

线材默认使用固定版本 [robosuite Panda 夹爪参数](../assets/reference/robosuite_panda/manifest.json)，每指独立位置执行器，刚度 1000 N/m、限幅 ±20 N，关节 damping=100、armature=1、frictionloss=1。Panda 的本地关节范围为 `[-0.04,0] m`，对应原每指 40 mm 行程；不保留人工 2/6 mm 空隙。两块指腹之间启用碰撞，实际开度由力学平衡决定。

Robotiq 使用 EBiM 原始 0–0.8 rad 关节范围、kp=300、kv=20、关节驱动扭矩上限 ±50 Nm；这些不是指尖接触力。新的软指腹安装不是原 EBiM 结构。原始行程下强压可翻转网格，当前不能宣称已稳定复现。

机械臂本地 PD 控制和 IK 不等于完整官方 Franka 控制器。`--grip-force` 显式启用的旧限力模式没有官方标定依据，不作为默认配置。

## 本地控制与失效检查

### 控制配置与对照入口

默认启用笛卡尔限制，包含键盘／SpaceMouse 每轴 6 mm 目标领先量，用于限制目标与实测位置之间积累的误差。各限制的具体作用及关闭方式见下文。

可自行运行 Panda cube 控制对照：

```bash
.venv/bin/python scripts/diagnostics/compare_cube_controls.py \
  --output outputs/diagnostics/my_control_comparison --seeds 0 1 2
```

脚本通过 `SpaceMouseController.action()` 输入处理链执行抓取、抬升、搬运和释放，比较限制开启／关闭两种配置。接触压力测试加 `--stress --no-record`，每次选择新输出目录。输入轨迹另存 `control_trace.npz`；报告和 RGB-D 仅保存在本地，不混入人工示范。这是控制与录制检查，不是学习策略评估。

### Robotiq 喂食初始化

`--gripper robotiq_2f85` 支持喂食。依据源 joint equality 同时初始化六个连杆关节；勺柄的初始位置取实际两侧指腹中点。预夹采用 [EBiM task3 update_gripper](https://github.com/EBiM-Benchmark/benchmark/blob/161db49e9eafd34be31e5bdada72d8e2f796c16a/task3_mujoco/teleop.py)：1 rad/s 闭合、两侧外部法向力之和低通 α=0.25、连续 5 个 1 ms 周期达到 24 N 后，保持实测转轴角＋0.005 rad；预夹目标上限 0.735 rad。数值读取原始 `assets/ebim_feeding/config.json`，函数源码与哈希保存在同目录。这里只把刚性指垫接触力提取替换为两侧柔性指腹的外部接触力总和，因此是有说明的仿真接入，不是原场景完全不变。

24 N 是预夹闭合过程的检测门槛，不是运行中的恒力指令或人脸力上限。仅 reset 使用该流程，之后保持位置；原生 0–0.8 rad 行程、kp=300、kv=20 和执行器限制保留。勺子靠接触与摩擦夹持，无焊接或附加托举力；再次下发全闭合仍可能导致指腹过压。

### 生效的控制与质量检查

没有 TCP 最低高度限制；桌面接触由求解器计算。默认仍保留以下本地控制设置，均不是论文规定，也不是传感器标定：

- 场景 XY 和最高 Z 工作范围：方块 X 0.24–0.58/Y ±0.18/Z≤0.40 m；线材 X 0.24–0.64/Y ±0.22/Z≤0.40 m；喂食使用其平移后的场景坐标范围。键盘、SpaceMouse、Pico 使用对应场景范围。
- 每控制步平移分量最多 5 mm、旋转向量长度最多 0.03 rad；IK 使用阻尼迭代、关节边界内侧 0.02 rad，手臂 PD 为 600/45 加偏置力补偿。实际执行器范围沿用模型。
- 键盘/SpaceMouse 的目标领先量为每轴 6 mm、旋转 0.03 rad；Pico 为每轴 30 mm、旋转 0.10 rad。速度限制和失焦/追踪失效停止用于输入控制，不用于伪造接触。
- 两侧指腹 `0<V/V0≤0.15` **只作网格质量警告**：正体积、求解正常且磁场有限时继续按当前形变计算，继续录制；没有保持最大值、复制上一帧、幅值截断或暂停。这一 15% 是本地未验证的诊断值，不是芯片量程、饱和阈值或材料极限。raw 的 `skin_min_volume_ratio`、`passive_min_volume_ratio`、`mesh_quality_warning` 与 100 Hz 时间戳同步，默认不因此过滤 episode。
- `V/V0≤0` 表示单元退化到零或发生翻转；非有限状态、磁场或 MuJoCo 求解发散也属于物理故障，仍终止该段有效采集并暂停。不会把故障前最后一帧重复写成新测量。网络后端没有可用预测时仍按其独立的缺失数据规则处理，与 15% 警告无关。
- 体积比针对单个四面体，取最小值；不等于整个皮肤厚度比，也不直接决定磁场大小。真实 AnySkin 没有网格阈值，测的是磁化皮肤变形引起的磁场变化。当前警告帧是可计算的仿真数据，但未验证的大变形结果不能当成准确真值。

`teleop --no-cartesian-limits` 可明确关闭工作空间、单步笛卡尔增量、目标速度平滑、目标领先截断。SpaceMouse/键盘回中后保留尚未到达的目标，会继续追赶；输入失焦／失联和 Pico 松握键仍取消目标。旋钮输入倍率、死区、Pico 追踪跳变检查、IK（8 次阻尼迭代、每次关节更新最多 0.04 rad、距关节边界 0.02 rad）、执行器力矩范围和接触物理仍在，不称为“完全无限制”。本地 PD 600/45、IK 的旋转误差权重 0.2／阻尼平方 0.0001 都是控制实现参数，未用真机辨识。

开启／关闭的实际区别：

- 工作空间开启时，目标在边界被截断，即使画面中还没有碰到东西也可能不再前进；关闭后目标可越过该盒子，但机器人关节可达性与碰撞仍然约束实际运动。
- 单步上限和目标速度平滑开启时，大输入被分步执行；关闭后 IK 直接追踪未作这层限速的目标，可能更快，也可能产生更大的跟踪误差和接触载荷。它不改变物理求解耗时，不能修复 0.01 倍实时的问题。
- 领先量限制开启时，目标只能领先实测位姿一小段，接触时可能感觉推不动；关闭后持续输入能积累较远目标，卸载后仍可能继续追赶。SpaceMouse 回中本身不会清掉无限制模式下的积累目标。
- 输入处理仍有本地设定：SpaceMouse 死区 0.08、归一化量程 350、默认速度 0.12 m/s／角速度 0.5 rad/s；每组非零轴数据超过 0.25 s 未更新会停止。Pico 超过 0.25 s 的数据、单帧位置跳变超过 0.15 m 或旋转超过 0.8 rad 会中断跟随；这些开关不随 `--no-cartesian-limits` 关闭。
- 网格 15% 警告与运动限幅无关；实际翻转、NaN 或求解器错误不能靠关掉诊断恢复有效物理。完整节点喂食路径还保留 CG 失败后恢复同一采样边界、改 Newton 重算的本地机制；降阶路径不执行这项切换。它没有补造测量，但也不是官方任务算法。

raw 保存的是处理后的动作；原始输入及逐次裁剪／重锚事件还未独立保存。因此不能声称已经具备完整的输入到执行审计。新增这些字段的剩余接入要求见 [TODO](../TODO.md)。

每次采集在 `metadata.json/controller_contract` 保存开关及实际生效值，`simulation_config` 保存输入倍率、物理和触觉配置。无上限字段为 null。两种模式不能混入同一采集目录；试验目录用 `data/episodes_unlimited/`。关闭这些限制不会给夹爪焊接物体、忽略碰撞或更改成功阈值。

回归入口：`tests.physics.test_contact_integrity` 检查网格加密、记录的抓绳/卡扣动作及释放、Panda 原行程空夹接触与两种夹爪的压桌/抬起；`tests.physics.test_reduced_skin` 检查三任务脚本。Robotiq 强夹持失败仍由 `tests.physics.test_robotiq` 暴露。通过少数轨迹不代表任意接触稳定，也不代表实物标定完成。

## 五个磁力计的位置

[ReSkin 原贴片坐标表](../assets/reference/reskin_pick_place.txt) 和 [sensor_layout.json](../assets/reference/sensor_layout.json) 提供 PCB 参考：中心一枚、四周约 7 mm 各一枚。仿真使用 U1中心/U2上/U3右/U4下/U5左；这是参考 PCB 丝印顺序，不应直接视为实际 WowSkin 串口顺序。

原厂可视化 CH1中心/CH2左/CH3右/CH4上/CH5下；本地显示适配从 U 顺序取 `[U1,U5,U3,U2,U4]` 并转换坐标轴。该约定可核对，但真机通道与轴向尚未全部验证。

已有 ReSkin 前向网络采用候选重排 `[0,1,3,2,4]`，由原数据组排列得到 CH 顺序；它没有解决未知三轴朝向。五点模长匹配也不能辨识轴向。

## 前向网络与数据

`outputs/reskin_forward_mapped/` 是本地小型前向网络，输入接触 XY/深度毫米，输出 15 路 ΔB（μT）。训练来源为 [3D Cal ReSkin 数据](https://zenodo.org/records/18462608)，实测深度为 1.5/2.0 mm；浅层、宽面、线接触及当前 WowSkin 均超出已验证范围。

`outputs/reskin_inverse_mapped/` 使用 MagNet 架构，由磁场反推位置/深度；这与仿真需要的前向方向相反。逆模型伪标签增加数据对，但不会新增独立测量的深度监督。

```bash
.venv/bin/python -m wowskin_lab reskin-response predict --xyz-mm 8 8 1.5
```

## Franka 安装与网格

左夹指安装 WowSkin，右夹指安装同尺寸普通硅胶；右侧只计算力学。原 Franka 黑色接触垫及其碰撞块已移除，组件上沿对齐安装面。安装位置来自原 OBJ，不是实物支架制造 CAD。

共用底座／硅胶套尺寸。降阶任务默认使用 48 个轮廓采样点；完整自由度任务仍为 12 个，独立传感器默认 48 个。降阶的形变自由度不随轮廓采样数增加，但碰撞和磁场积分的计算量会增加。局部体积比依赖网格离散，不应直接解释为实物硅胶的体积变化；完整的载荷、接触和磁场收敛检查仍待完成。皮肤受力、接触法向力和磁场分别保存，不能把五个磁场点视为五个测力单元。

## 真机与仿真显示

```bash
.venv/bin/mjpython -m wowskin_lab sensor viewer --ideal --real-port auto --capture-dir outputs/calibration
```

自动选择唯一可识别串口，启动前保持传感器固定、无接触。界面 `REAL on model` 在 MuJoCo 背景上显示真实串口数据；按 T 切换 `SIM prediction` 比较独立仿真。真实手按不会自动驱动仿真形变。

- 1–5 选择中心/左/右/上/下；P 仿真按压，L 抬起。
- B 归零真机，Z 归零模拟显示；Reset 取接下来 5 帧平均并固定减去。
- M 保存时间标记与截图；Q 结束并保存。
- `real.jsonl`、`sim.jsonl`、`events.jsonl`、`serial.bin`、`manifest.json` 保存原始读数、基线、事件及配置。

原厂绘制源码在 [wowskin_visualizer](../third_party/wowskin_visualizer/wowskin_viz.py)，默认 7 μT/逻辑像素。绿色为 XY，红色 Z 正值实心、负值空心。主机接收时间关联不等于硬件同步；图形一致不等于仿真已标定。

## 能用真实 AnySkin/WowSkin 来标定吗

归零只确定参考磁场；手按可以核对通道、空载波动、重复性与卸载恢复。

要辨识前向响应，使用已知形状的非磁性压头和可读位移的加载支架，在已知位置做多深度加载、保持、卸载。记录原始 15 路磁场、时间戳、基线、压头形状、实际压入量和安装方向。若要辨识刚度或牛顿单位力，还需同步参考测力。

先核对位置、通道与轴，再拟合少量磁场参数，然后结合力—位移拟合力学/动态参数；留出完整位置和加载过程检验。单凭未测载荷的手按曲线或 Pollen 滑移标签，不能唯一辨识材料参数。

## 触觉仿真与标定文献

相关方法均尚未接入本项目；文献中的保证不能直接归给当前 MuJoCo 后端。

- **TacIPC**（RA-L 2024）：[论文](https://arxiv.org/abs/2311.05843)、[代码](https://github.com/Ailon-Island/TacIPC)。FEM＋IPC 弹性层求解，防接触穿透和单元翻转；可借鉴指腹材料、固定边界与接触算法，输出形变后仍需另建磁场模块。
- **TACTO**（RA-L / ICRA 2022）：[论文](https://arxiv.org/abs/2012.08456)、[代码](https://github.com/facebookresearch/tacto)。物理引擎与光学触觉渲染分工，生成 DIGIT / OmniTact 图像；适合借鉴软件分层，不负责精确 FEM 接触求解。
- **DiffTactile**（ICLR 2024）：[论文](https://arxiv.org/abs/2403.08716)、[项目](https://difftactile.github.io/)、[代码](https://github.com/Genesis-Embodied-AI/DiffTactile)。可微 FEM 与多材料接触，包含线材任务和参数辨识；可借鉴实测轨迹拟合、留出验证，但光学观测换成磁场需要新增观测模型。
- **3D Cal / ReSkin 扩展**：[论文](https://arxiv.org/abs/2511.03078)、[ReSkin 说明](https://rohankotanu.github.io/3DCal/reskin.html)、[代码](https://github.com/rohankotanu/py3DCal)、[数据及权重](https://zenodo.org/records/18462608)。已知 XY / 压入深度采集磁场，官方 MagNet 是 15 路磁场→位置/深度的逆模型。这里的前向 MLP 是本地训练的模型，不是官方前向模型；线材与 WowSkin 迁移尚未验证。

## Mac 上可借鉴的触觉求解器

- **TacIPC 原版**：[源码](https://github.com/Ailon-Island/TacIPC)、[构建依赖](https://github.com/Ailon-Island/TacIPC/blob/main/CMakeLists.txt)。项目声明 CUDA 语言并要求 CUDA Toolkit，不能直接在 Apple Silicon 上编译。Metal 是另一套 GPU 计算 API；把 CUDA 换成 Metal/MPS 需要移植求解器内核和依赖，不能通过安装一个替代库完成。本项目尚未接入 TacIPC。
- **CPU IPC 路线**：[IPC](https://github.com/ipc-sim/IPC) 的官方文档包含 macOS 构建依赖，可作为 FEM＋接触防穿透、防翻转的参考，无需 CUDA。此路线尚未在本机编译或测量速度，不承诺实时。
- **IPC Toolkit / PolyFEM**：[IPC Toolkit](https://github.com/ipc-sim/ipc-toolkit) 提供屏障能量、连续碰撞检测、摩擦及导数；它本身没有材料物理或完整求解器。官方建议结合 [PolyFEM](https://polyfem.github.io/)。接入时仍须实现传感器材料、边界与形变到磁场的模型。
- **TACTO**：[源码与兼容说明](https://github.com/facebookresearch/tacto)。Python＋PyBullet＋Pyrender，不强制依赖 CUDA；官方推荐 Ubuntu，并指出 macOS 的 PyBullet GUI 和 Pyrender 显示兼容问题。因此可以尝试 Mac 运行，但尚未在本机安装验证。它生成 DIGIT / OmniTact 等光学触觉的 RGB / 深度，不直接生成 AnySkin 的 15 路磁场，也不能替换硅胶 FEM 接触求解。

这些都是候选参考，并非当前运行后端。替换形变求解器后，磁场响应与实物标定仍需要单独处理。
