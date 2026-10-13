# 传感器模型与标定

本项目的 WowSkin 是本地物理近似模型，尚未完成实物材料与磁场标定。启动与录制见 [README](../README.md) 和 [操作说明](COLLECTION_ZH.md)。

## 计算方式来自哪里

AnySkin 的磁化皮肤受接触后变形，改变下方五个三轴磁力计的读数。[AnySkin 论文](https://arxiv.org/abs/2409.08276)

本项目用 MuJoCo 弹性四面体网格近似皮肤形变，再选择磁场输出方式；它不是 AnySkin、ViSk 或 FTF 官方提供的传感器仿真。三个方式共享同一套接触力学：

- **原偶极子（底层参考）**：每个四面体中心放置磁偶极子，位置随网格移动，方向随形变梯度的旋转部分改变；叠加五芯片处磁场并扣除基线。遥操作入口不默认选择此项。
- **拟合偶极子（各任务的遥操作默认）**：使用 `configs/magnetic_fit_pollen_nonslip.json`，在偶极子场上应用五个正幅值系数。运行时不查询网络；也可用 `--magnetic-fit` 显式指定配置。
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
- 材料：杨氏模量 30 kPa、泊松比 0.35、密度 1000 kg/m³、摩擦系数 0.6；弹性阻尼 1e-5。机器人上完整节点模型使用局部节点阻尼 0.002，trilinear 降阶节点关节阻尼为 0；独立传感器台（包括默认 trilinear）在 `physics.py` 中统一设置自由度阻尼 0.002，三平移自由度探头另设为 0.02。这些是未辨识的本地初值，不是厂商参数或经过验证的经验值。
- 磁场：均匀 +Z 磁化 4000 A/m、芯片至软层底面间距 2.2 mm，均未实测辨识。
- 信号：100 Hz；独立高斯噪声标准差 0.5 μT、一阶滤波时间常数 10 ms、延迟 1 个采样。噪声不累加，线性漂移默认关闭。
- 标签：接触阈值 0.005 N、滑移速度阈值 0.0005 m/s，为本地标签规则，不是 Pollen 的人工标注。
- 数值：默认 trilinear 降阶使用 MuJoCo 3.15、discrete、CG、500 μs，不启用内部接触。完整网格对照使用 MuJoCo 3.5；其线材/喂食启用内部接触。线材接触半径 0.2 mm、时间常数 0.2 ms、`solimp=0.98 0.98 0.001 0.5 2` 都是本地数值设置，不是实测材料。尚无完整收敛验证。[MuJoCo 内部接触说明](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html#flex-contact)

配置：[sensor.json](../configs/sensor.json)、[sensor_hardware.json](../configs/sensor_hardware.json)。密度 1000 kg/m³ 和弹性阻尼 1e-5 硬编码于 [geometry.py](../wowskin_lab/core/geometry.py)，不在上述 JSON 中；独立台数值阻尼见 [physics.py](../wowskin_lab/core/physics.py)。

## 夹爪配置来源

[gripper_control.json](../configs/gripper_control.json) 使用固定版本 Google DeepMind Menagerie Panda 位置执行器，来源与哈希在 [manifest](../assets/reference/panda_menagerie/manifest.json)。原位置增益 100 N/m、速度增益 10 N·s/m、tendon 限幅 ±100 N，两指传动各乘 0.5。

本地关节坐标 `q_local=q_source−0.04`，闭合指令方向相反；实际 tendon 输出为 `−4×closure−100×q_local−10×qdot`。因此 ±100 N 不是恒定夹紧力。新增软指腹的接触负载仍可能顶开位置伺服；不能用磁场倍率解决该问题。

线材默认使用固定版本 [robosuite Panda 夹爪参数](../assets/reference/robosuite_panda/manifest.json)，每指独立位置执行器，刚度 1000 N/m、限幅 ±20 N，关节 damping=100、armature=1、frictionloss=1。Panda 的本地关节范围为 `[-0.04,0] m`，对应原每指 40 mm 行程；不保留人工 2/6 mm 空隙。两块指腹之间启用碰撞，实际开度由力学平衡决定。

Robotiq 使用 EBiM 原始 0–0.8 rad 关节范围、kp=300、kv=20、关节驱动扭矩上限 ±50 Nm；这些不是指尖接触力。新的软指腹安装不是原 EBiM 结构。原始行程下强压可翻转网格，当前不能宣称已稳定复现。

### 指腹与底座接触

显式 `--pad-backing-contact`（默认关闭）记录 `pad_backing_contact=true`，恢复硅胶与硬底座的碰撞：指腹 `contype=2, conaffinity=6`，硬底座由 `1/5` 改为 `1/7`。旧掩码同时排除了指腹对自己及对面底座的碰撞。三线性模型只固定包围盒底面节点，不能替代硅胶下方实际硬底座的支撑。此修正不改变材料、夹爪目标、增益、力上限、原始关节行程或磁场输出，不增加固定闭合空隙。旧记录不含此键时按旧碰撞配置重建；对照采集必须使用新目录。它可缓解空夹压入底座，但增加约百个空载接触，当前明显拖慢求解，强夹薄线仍会翻转；因此尚未替换默认模型。

这还不等于完成大应变材料验证：当前固定面仍是降阶近似，`young=30 kPa`、`poisson=0.35` 未标定，网格、接触和时间步收敛尚待检查。[MuJoCo 弹性模型说明](https://mujoco.readthedocs.io/en/latest/XMLreference.html#flex-elasticity)指出其适用的小应变范围。单元体积比降到零／负值，代表数值网格塌缩／翻转，不是传感器达到最大读数。

`--finger-contact-stop` 是独立、默认关闭的操作选项：检测实际指腹对指腹的接触，在还要求继续闭合时锁定当时实测闭合比例；持续 CLOSE 不继续增加目标，张开指令解除。对物体／桌面的接触不触发这项停止，物理和触觉始终继续计算。启用时界面显示 `Contact stop ON`，触发时显示 `HOLD: pads touched`。raw 分别记录 `requested_gripper`（该步请求）、`commanded_gripper`（实际目标）和 `finger_contact_stop_active`；训练仍使用实际目标。它限制过压工况，不能替代模型修正，也不能保证其他压碰不会翻转。

机械臂本地 PD 控制和 IK 不等于完整官方 Franka 控制器。`--grip-force` 显式启用的旧限力模式没有官方标定依据，不作为默认配置。

## 本地控制与失效检查

### 控制配置与对照入口

默认遥操作使用机器人基座方向，本地桌面工作空间盒子关闭。仍启用速度、单步增量和目标领先限制；键盘／SpaceMouse 每轴 6 mm 领先量用于限制目标与实测位置之间积累的误差。各限制的具体作用及关闭方式见下文。

可自行运行 Panda cube 控制对照：

```bash
.venv/bin/python scripts/diagnostics/compare_cube_controls.py \
  --output outputs/diagnostics/my_control_comparison --seeds 0 1 2
```

脚本通过 `SpaceMouseController.action()` 输入处理链执行抓取、抬升、搬运和释放，比较限制开启／关闭两种配置。接触压力测试加 `--stress --no-record`，每次选择新输出目录。输入轨迹另存 `control_trace.npz`；报告和 RGB-D 仅保存在本地，不混入人工示范。这是控制与录制检查，不是学习策略评估。

Peg 的关节 PD／笛卡尔阻抗对照入口：

```bash
.venv/bin/python scripts/diagnostics/compare_peg_controllers.py \
  --output outputs/diagnostics/my_peg_controllers
```

通过与遥操作相同的控制入口，读取仓库内固定版本 `assets/reference/serl_controller/impedance.py`。默认比较自由移动、对准插入、横向偏置 3 mm 三种情况。遥操作也可用 `--arm-controller cartesian_impedance` 显式选择，默认仍为 IK＋关节 PD；两边使用相同的物理夹持、目标平滑、执行器范围、碰撞与成功条件。初始化先以同一关节 PD 夹住 peg，再开始对照。插入使用同一物体位姿反馈脚本，不是学习策略。插入默认沿用键盘／SpaceMouse 每轴 6 mm 目标领先限制；`--target-lead-m 0` 单独测目标持续累积的工况。自由移动按相同的时间轨迹直接给目标，不施加这项领先截断，以比较实际跟踪误差。

笛卡尔分支直接执行 SERL++ 控制函数，保留默认增益和算法，只适配 MuJoCo 3.15 的 `mj_fullM` 参数接口；零空间参考设为本场景初始关节姿态。源文件摘要和具体接口改动写入报告。输出包含 TCP 跟踪误差、插入评分、每个物理步的 peg／孔座接触力峰值及运行耗时；法向力之和与合力范数分别记录。整段耗时包含逐物理步的接触统计与各自控制器计算（笛卡尔分支不运行 IK），不等于遥操作 GUI 帧率。两套默认增益没有匹配等效刚度，也未调优，不能据此断言某一类控制器普遍更准。力值来自未标定的本地接触模型；3 mm 偏置超过评分允许的 2 mm，用于观察孔口接触，不能作随机扰动成功率测试。

### 卡扣的物理范围

默认 `--clip-model rigid` 是刚性开口碰撞代理，不模拟弯曲或回弹。可选 `--clip-model elastic_beam` 使用下面的实验模型。两者都不在进入区域后额外吸附线材。EBiM 固定版本的 XML 注释曾描述 `weld`，但实际 [apply_clip_guide](https://github.com/EBiM-Benchmark/benchmark/blob/161db49e9eafd34be31e5bdada72d8e2f796c16a/task1_mujoco/robotiq_duo_full_scene_minimal_core/teleop/grasping.py) 是向槽中心施加横向弹簧阻尼力，单个槽内线段限幅 8 N，允许沿槽轴滑动；本地没有接入这项辅助力。它也不是弹性卡扣本体求解。

若研究真实扣合触觉，需要考虑夹臂／唇边被撑开后回弹，以及线材截面受压，不能只扩大成功区域。可参考 [Mechanics of a snap-fit](https://arxiv.org/abs/2003.13566) 的弹性壳弯曲、接触摩擦和插拔力实验；[BrickSim](https://intelligent-control-lab.github.io/BrickSim/) 则采用刚体引擎与专门连接力模型的混合方法，研究的是积木连接，不是本卡扣的现成模型。TacIPC 研究弹性触觉指腹，其 FEM／接触求解可作为方法参考，未提供此 C-Clip 的标定模型。

### 实验弹性卡扣

入口为 `teleop --task cable --cable-layout single_clip --clip-model elastic_beam`，也支持 `routing`、Panda／Robotiq 和两种控制器。原 EBiM STL 的安装底座保持固定；上壁、弯角、上唇组成一条被动梁链，下唇是独立梁链，两条根部都连接在固定底座上。可见表面和碰撞表面一起变形，不是只改渲染。关节弯曲能量为 `E = ½ kθ θ²`，刚度由两相邻半段的弯曲柔度计算。线材或压头的碰撞力撑开夹臂，撤去接触后靠弹簧势能回弹；没有吸附、自动闭合或进入成功框后施加辅助力。上下唇长短不同，使用同一材料也不会产生相同位移。

这是参考 [Mechanics of a snap-fit 的离散弹性模型](https://arxiv.org/html/2003.13566v1#A3) 的本地**平面、不可伸长梁近似**，并未复现论文全部模型。没有三维扭转或完整壳体 FEM。几何截面近似、杨氏模量 1 GPa、密度 1000 kg/m³ 和阻尼都未标定；参数来源逐项保存在 [elastic_clip.json](../configs/elastic_clip.json)。适用当前 MuJoCo 3.15 discrete 降阶指腹配置，其他积分器尚未验证。

`check_elastic_clip.py` 用运动压头验证撑开／撤回／回弹。机器人夹线插入／松手保持另由 `tests.physics.test_elastic_clip` 检查。压头测试的接触力峰值尚未时间步收敛，不能用于定量插入力结论。成功评分仍沿用几何入槽条件，不能把 SUCCESS 当作已验证弹性锁定。回归入口：

```bash
.venv/bin/python scripts/diagnostics/check_elastic_clip.py --output outputs/diagnostics/my_elastic_clip
.venv/bin/python scripts/diagnostics/check_elastic_clip.py --probe-height-mm 7 --output outputs/diagnostics/my_lower_lip
```

### Robotiq 喂食初始化

`--gripper robotiq_2f85` 支持喂食。依据源 joint equality 同时初始化六个连杆关节；勺柄的初始位置取实际两侧指腹中点。预夹采用 [EBiM task3 update_gripper](https://github.com/EBiM-Benchmark/benchmark/blob/161db49e9eafd34be31e5bdada72d8e2f796c16a/task3_mujoco/teleop.py)：1 rad/s 闭合、两侧外部法向力之和低通 α=0.25、连续 5 个 1 ms 周期达到 24 N 后，保持实测转轴角＋0.005 rad；预夹目标上限 0.735 rad。数值读取原始 `assets/ebim_feeding/config.json`，函数源码与哈希保存在同目录。这里只把刚性指垫接触力提取替换为两侧柔性指腹的外部接触力总和，因此是有说明的仿真接入，不是原场景完全不变。

24 N 是预夹闭合过程的检测门槛，不是运行中的恒力指令或人脸力上限。仅 reset 使用该流程，之后保持位置；原生 0–0.8 rad 行程、kp=300、kv=20 和执行器限制保留。勺子靠接触与摩擦夹持，无焊接或附加托举力；再次下发全闭合仍可能导致指腹过压。

### 生效的控制与质量检查

没有 TCP 最低高度限制；桌面接触由求解器计算。默认仍保留以下本地控制设置，均不是论文规定，也不是传感器标定：

- **工作空间默认关闭**：只有显式 `--workspace-limits` 才启用旧本地世界坐标盒子：方块 X 0.24–0.58/Y ±0.18/Z≤0.40 m，线材 X 0.24–0.64/Y ±0.22/Z≤0.40 m；喂食使用独立硬编码的世界坐标盒子 X −2.01–−1.26/Y −0.36–0.52/Z≤1.68 m，`care_room` 将 X 上限改为 −1.0 m；该范围不是由方块盒子按基座位姿变换得到。H 可查看实际范围。不是厂家可达空间，也不是碰撞边界。
- **输入方向固定**：三种输入默认 `--control-frame base`；W/S 对应基座 ±X，A/D 对应 −Y/+Y，E/Q 对应 ±Z。SpaceMouse 平移与旋转参照固定基座，键盘旋转键绕 TCP 局部轴；Pico 的佩戴／对面模式校准用户方向后映射到基座，不参考预览相机。`--control-frame view` 才选择旧视角映射。
- 每控制步平移分量最多 5 mm、旋转向量长度最多 0.03 rad；IK 使用阻尼迭代、关节边界内侧 0.02 rad，手臂 PD 为 600/45 加偏置力补偿。实际执行器范围沿用模型。
- **领先量**：键盘/SpaceMouse 为每轴 6 mm、旋转 0.03 rad；Pico 为每轴 30 mm、旋转 0.10 rad。平移到边界时统一缩短整步，不分别裁剪各轴，也不把横向跟踪误差写回移动中的目标。接触时仍会限制进一步积累。
- **输入速度和旋转**：键盘方块 0.04 m/s、其他任务 0.12 m/s，角速度 0.4 rad/s；SpaceMouse 0.12 m/s、0.5 rad/s；Pico 0.25 m/s、1.5 rad/s。底层目标再受 `tcp_speed_m_s`／`tcp_angular_speed_rad_s` 约束，实际值写入 metadata。均为本地操作参数，不是厂家最大速度。
- **松手／回中处理**：默认键盘释放、SpaceMouse 回中时把目标重新设到实测位姿，取消尚未追完的运动；失焦／失联也取消。Pico 松握键停止，再握住建立新参考；模式切换与校准也重新建立操作参考。这些行为会影响操作手感。
- 两侧指腹 `0<V/V0≤0.15` **只作网格质量警告**：正体积、求解正常且磁场有限时继续按当前形变计算，继续录制；没有保持最大值、复制上一帧、幅值截断或暂停。这一 15% 是本地未验证的诊断值，不是芯片量程、饱和阈值或材料极限。raw 的 `skin_min_volume_ratio`、`passive_min_volume_ratio`、`mesh_quality_warning` 与 100 Hz 时间戳同步，默认不因此过滤 episode。
- `V/V0≤0` 表示单元退化到零或发生翻转；非有限状态、磁场或 MuJoCo 求解发散也属于物理故障，仍终止该段有效采集并暂停。不会把故障前最后一帧重复写成新测量。网络后端没有可用预测时仍按其独立的缺失数据规则处理，与 15% 警告无关。
- 体积比针对单个四面体，取最小值；不等于整个皮肤厚度比，也不直接决定磁场大小。真实 AnySkin 没有网格阈值，测的是磁化皮肤变形引起的磁场变化。当前警告帧是可计算的仿真数据，但未验证的大变形结果不能当成准确真值。

`teleop --no-cartesian-limits` 可明确关闭工作空间、单步笛卡尔增量、目标速度平滑、目标领先截断。SpaceMouse/键盘回中后保留尚未到达的目标，会继续追赶；输入失焦／失联和 Pico 松握键仍取消目标。旋钮输入倍率、死区、Pico 追踪跳变检查、IK（8 次阻尼迭代、每次关节更新最多 0.04 rad、距关节边界 0.02 rad）、执行器力矩范围和接触物理仍在，不称为“完全无限制”。本地 PD 600/45、IK 的旋转误差权重 0.2／阻尼平方 0.0001 都是控制实现参数，未用真机辨识。

开启／关闭的实际区别：

- 显式开启工作空间时，请求的整步在边界缩短或停止，即使画面中还没有碰到东西也可能不再前进；关闭后目标可越过该盒子，但机器人关节可达性与碰撞仍然约束实际运动。
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

当前 MuJoCo 3.15 的独立传感器入口也使用 trilinear＋discrete；历史 3.5 完整节点按压曲线不作为这一模型的标定结果。配置会写入采集记录。

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

论文用途、公开代码和本项目接入状态集中在[论文与算法索引](PAPERS_ZH.md)。下面只保留求解器选择和本地模型的具体适配说明；引用论文不表示本项目已获得其精度或物理保证。

## Mac 上可借鉴的触觉求解器

- **TacIPC 原版**：[源码](https://github.com/Ailon-Island/TacIPC)、[构建依赖](https://github.com/Ailon-Island/TacIPC/blob/main/CMakeLists.txt)。项目声明 CUDA 语言并要求 CUDA Toolkit，不能直接在 Apple Silicon 上编译。Metal 是另一套 GPU 计算 API；把 CUDA 换成 Metal/MPS 需要移植求解器内核和依赖，不能通过安装一个替代库完成。本项目尚未接入 TacIPC。
- **CPU IPC 路线**：[IPC](https://github.com/ipc-sim/IPC) 的官方文档包含 macOS 构建依赖，可作为 FEM＋接触防穿透、防翻转的参考，无需 CUDA。此路线尚未在本机编译或测量速度，不承诺实时。
- **IPC Toolkit / PolyFEM**：[IPC Toolkit](https://github.com/ipc-sim/ipc-toolkit) 提供屏障能量、连续碰撞检测、摩擦及导数；它本身没有材料物理或完整求解器。官方建议结合 [PolyFEM](https://polyfem.github.io/)。接入时仍须实现传感器材料、边界与形变到磁场的模型。
- **TACTO**：[源码与兼容说明](https://github.com/facebookresearch/tacto)。原版使用 PyBullet＋Pyrender，不强制依赖 CUDA，原 GUI 有 macOS 兼容限制。本项目已接入固定版 MuJoCo 移植与 CGL 预览，见下文 DIGIT / TACTO 选项；不等于原 PyBullet 整套示例均已验证。其 RGB / 深度不直接生成 AnySkin 的 15 路磁场，也不能替换硅胶 FEM 接触求解。

TacIPC 和 CPU IPC 路线仍是候选；TACTO 已接入 Panda 光学显示与录制。替换形变求解器后，磁场响应与实物标定仍需要单独处理。

## 脆性薯片预分块试验

MuJoCo 没有设置“玻璃材质”就自动产生裂纹和碎片的内置模型。其维护者给出的实现路线是：把物体预先拆成多个刚体，用 weld 连接，在力或位移超过规定条件后解除连接；当前 API 支持通过 `mjData.eq_active` 关闭对应约束。[维护者说明](https://github.com/google-deepmind/mujoco/issues/577)、[weld 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld)。该链接中的断裂判据只是示例，不能直接当作玻璃材料参数。

`python -m wowskin_lab brittle-chip` 已提供独立薯片按压台，Mac 窗口使用 `mjpython`。实现位于 `wowskin_lab/envs/brittle_chip.py`，所有试验配置在 `configs/brittle_chip.json`。默认外形为约 70×50×1 mm 的弯曲薄片，密度 450 kg/m³，24 个三角棱柱组成 32 条共享边连接；多边形离散后质量约 1.11 g。**尺寸、密度、接触摩擦和断裂强度都是本地试验设定，尚无实物标定依据。**

- 每条连接独立计算共享边中点处的合力与合力矩，默认达到 0.3 N 或 0.0005 N·m 就不可逆断开。判据使用范数，不区分拉／压／剪切模式，且依赖分块方式；不能称为材料应力模型。`efc_force` 的旋转行不能直接当成 N·m，代码通过约束 Jacobian 转成广义力，再换算到连接中点的世界系力矩。
- weld 采用 MuJoCo 默认 1 m `torquescale`，配置中较高的约束阻抗用于近似刚性连接；这些是数值设置，不是薯片杨氏模量。默认时间步 1 ms、`solref` 时间常数 3 ms。修改时间步／分块／约束设置后需要重新验证，不能假定任意设定都给出相同破坏力。
- 初始化按碰撞几何把完整薄片放到两个支架上；之后没有固定薄片的隐藏 weld。压头是位置伺服驱动的独立滑动刚体，外力来自真实接触。断裂只关闭对应 `eq_active`，不改碎片位置／速度，也不注入弹飞速度；只有用户 Reset 才恢复连接。
- 同一连通刚体簇的内部拼接面不参与自碰撞，否则会把物体内部当作外表面互相挤压。代码显式更新碰撞掩码；一旦分成不同连通簇，簇之间立即恢复碰撞。压头、支架和地面的碰撞始终开启，没有全局关闭碎片碰撞。
- 当前是预定义裂缝的刚体模型，没有连续材料应力、动态生成裂纹、断裂能或碎屑粉化；几何与强度也不能直接用于玻璃。新增 `teleop --task chip_tray` 接入 Franka／AnySkin，原有任务保持不变。该实验暂用人工结果标签，未定义自动无损夹取成功阈值。

机器人版的 [chip_tray.json](../configs/chip_tray.json) 覆盖马鞍幅度为 12 mm（边缘高低差 24 mm），4 内圈点／16 外圈点形成 24 块、28 个连接；碰撞和渲染共用实际薄片网格。托盘沿用 EBiM 的网格、质量、惯量和接触参数，初始化高度调整到桌面，外观使用源场景碗的 `mat_v2_OmniPBR` 蓝绿色材质。碎片使用高位碰撞掩码，保留机器人／指腹原有低位碰撞层。AnySkin 模式下，稀疏 CG 迭代上限为 100、容差为 1e-10；30 次迭代不足以解析这些刚性连接。降阶指腹仍使用 discrete 积分，不切换到不兼容的 Newton。显式选择光学传感器后，最终装配会用光学求解器配置整体替换任务的 `option` 设置；例如 TACTO 改为 Newton／100 与 implicitfast，以上 CG／discrete 描述仅适用于 AnySkin。初始化沉降期间同样计算断裂，不暂停检测或提高强度。

检查入口：`python -m unittest tests.physics.test_brittle_chip tests.physics.test_chip_tray`；涵盖空载保持、受力断裂、加大强度后保持、断裂时状态连续、碎片碰撞、重置及已知载荷下的力矩单位。可用 README 中的无界面命令生成本机视频与每条断键事件。

**NeoSim `grasp_chip` 的区别**：核对版本 `ad2d9b1ead7274391b348641e9abf38167e04f1e`，[物体生成脚本](https://github.com/neoteai/N0-Foundation/blob/ad2d9b1ead7274391b348641e9abf38167e04f1e/neosim/scripts/asset_tools/build_chip.py) 将 82.5×18×2.4 mm 长方形薄片沿长度方向弯曲，两端比中间高约 13.5 mm；它不是本项目的双曲率马鞍薯片，源码也未将它定义为电子芯片。[任务代码](https://github.com/neoteai/N0-Foundation/blob/ad2d9b1ead7274391b348641e9abf38167e04f1e/neosim/envs/grasp_chip.py) 使用刚性薄片，以触觉深度代理量 `far_plane - min_depth` 超过 8.5 mm 判失败，不执行几何断裂。这是任务损伤判据，不是以 N 为单位的断裂力，不能直接作为 AnySkin 的阈值。

本项目的 `bond_force_limit_n`、`bond_moment_limit_nm` 都可调整，任意一个连接达到力或力矩阈值便断开；提高力阈值不会消除力矩触发。它们目前只是试验强度，没有实物依据。更换成工业薄片也仍需明确材料和破坏判据，物体名称不能提供这些参数。

### 薯片夹持力反馈

`chip_tray` 的遥操作闭合命令启用 `core/grip_force.py` 中的导纳外环；它按上一物理步的 MuJoCo 接触力更新指关节位置目标，再交给原 Menagerie 位置伺服执行。设两指全部接触的法向力总和为 `F_L,F_R`，则 `F=max(F_L,F_R)`，`q_target_dot=-K*(F_target-F)`。负号对应 Panda 本地关节坐标下的闭合方向。单个耦合驱动无法分别控制两指；选较大值避免平均值掩盖单侧过载，双侧不平衡仍应检查抓姿。

外环不读取物体类别、宽度、位置、质量、破碎状态或成功标签。反馈包含托盘等所有与指腹接触的物体；它不是 AnySkin 15 路磁场的反演力。默认目标 0.05 N、增益 0.1 m/(N·s)、每指目标速度上限 10 mm/s 都是显式实验控制设置；空载、默认目标下每指目标速度为 5 mm/s，接近目标力时自动减速。保留原关节行程和执行器限幅，没有增加夹持宽度截断。更换目标力通过 CLI、按键或 API 完成，不改断裂强度。若受力超过目标，外环会退让；若低于目标则继续闭合。没有接触时会继续寻找接触，直到原生行程边界。

目标力是反馈设定值，不是瞬时接触力的硬上限；碰撞仍可能超调。两指互相接触也会产生读数，不能把“力达到目标”单独当作夹住薯片或成功抬起的证据。抬升测试另查薯片质心高度和与薯片本身的接触；这些评估量不进入控制器。

可用 `.venv/bin/python -m scripts.diagnostics.check_chip_force --forces .05 .08 .10 --closing-seconds 12` 对照不同目标力。输出包含薯片质量、质心高度、断裂事件，以及每指作用在薯片上的世界系法向力／摩擦力向量和其它外部接触合力。由此区分“压在上表面”“侧面夹持”和“脱落后两指互碰”；`--grasp-z`、`--yaw-deg` 是显式测试抓姿，不输入力控制器。世界系 Z 向上，静止时全部外部接触力的 Z 分量应近似平衡重力。

`--offset-xy DX DY` 可测试偏向边缘的抓姿；`--chip-frame-height H` 则先物理沉降 1.5 秒，使用仿真中的薯片位姿规划对照抓姿，替代固定世界系抓点。这属于已知物体几何的诊断，不能作为视觉策略成功的证据，也不改变在线遥操作或力控制器输入。`--snapshots` 保存闭合前、闭合后和抬升后的实际 MuJoCo integration state 与场景 XML，方便检查碰撞接触；遇到断裂时快照可能是提前终止状态，必须结合报告中的 `first_break` 阅读。

原 7D `step` 的最后一维仍是内层伺服闭合指令；程序化力控需显式 `env.command_gripper(True)`，之后用 `env.step(env.hold_action())` 保持外环。`env.command_gripper(False)` 停止力控并打开。raw 单独保存 `gripper_force_feedback/{enabled,target_n,selected_target_n,pad_normal_force_n,width_m,servo_target_width_m,servo_target_closure}`，不会把牛顿伪装成已有训练接口的开度。`width_m` 等于两个 Panda 本地指关节位置各加 0.04 m 后求和，`servo_target_width_m` 使用同样变换；实测值不裁剪，不扣软指腹厚度或形变。这是源夹爪编码器定义的开口，不是接触面距离。此模式当前只接入薯片 Panda 场景，不改变其它任务或已有 ViSk checkpoint 的控制语义。现有导出没有将最后一维改成目标力；当前 ViSk 方向保持触觉输入、运动与开合目标输出，不计划为此增加力 action。

薯片夹持仍待调整，未验证稳定无损抬升。后续接触与断裂检查统一列在 [TODO](../TODO.md#薯片夹持待调整暂不作为训练主线)，不通过提高默认破坏阈值或附着物体掩盖问题。

## 传感器硬件与光学算法

硬件和算法分开选择：`--sensor digit --optics tacto|fots`、`--sensor gelsight`。`digit-compare` 是独立比较工具。旧平面代理、正交深度、自定义材料点重锚与多物体 marker 叠加均已移除。

DIGIT 两条路径分别来自作者的不同模型，不强行共用 CAD、接触参数或相机。光学配置只读取 `configs/optical.json` 与 Panda 控制配置，**不读取 AnySkin 材料／磁场参数**。作者源码、权重与 CAD 分开下载，版本及文件 SHA256 见 `assets/reference/tacto/manifest.json` 和 `assets/reference/fots/manifest.json`。

**安装与运行适配，以下不是厂家硬件设计：**

- 保留 Panda 手掌、两指滑动关节与行程。TACTO 是完整长指组件，GelSightMujoco 是带长安装尾部的完整外壳，两者替换原短指几何与惯量；FOTS 是独立模块，保留原指并在末端固定连接。源模块质量与惯量计入运动链。名义全开时两侧源接触参考平面相距 80 mm，曲面胶层各处的可视间距不等于这个常数；TCP 移到参考平面之间。没有裁剪支架、缩放壳体、重造胶面或用包围盒估计惯量。此装配尚未制造实物。
- 安装轴由两份源 CAD 分别确定：TACTO 的圆弧端是源 −Z，FOTS 的圆弧端是源 −X；均映射到 Panda finger 的 +Z（远离手掌）。TACTO 源 +Y、FOTS 源 +Z 的感应面法向均映射到 finger −Y，两指相向。这是本地刚性安装变换，不能称为来源仓库的 Franka 支架设计；相机与胶面内部相对位姿不变。
- 名称加前缀，碰撞 bit 映射到任务场景；有效接触参数先在隔离模型编译，再明确写入，避免继承 Panda 或 AnySkin 默认值。原相机随完整本体移动。
- Panda 控制仍为项目的 IK／关节 PD 与任务原夹爪位置伺服（方块／喂食／薯片为 Menagerie，线材／Peg 为 robosuite 配置）。机器人场景保留重力 −9.81 m/s²；不沿用 FOTS 无重力按压装置或 TACTO 按压装置的重力补偿。
- 运行时间步为 0.5 ms。独立诊断探头为 5 mm 球或相近尺度方体／圆柱、20 g、位置伺服 kp=3000、kv=2√(3000×0.02)−1≈14.492，另有 joint damping=1 N·s/m，合计为临界阻尼；该小惯量六轴装置用 implicitfast 积分，属于测试夹具，不是传感器参数。Panda 的源求解器选择不被该夹具覆盖。临界阻尼用于限制测试伺服超调；球／方体／圆柱按压和卸载由独立测试覆盖，不修改传感器薄层或关闭外壳碰撞。
- `cable --clip-model elastic_beam` 是明确的任务积分例外：原积分器显式处理高刚度卡扣关节弹簧时在 0.5 ms 步长发散，改用 MuJoCo `discrete` 积分。保持来源 Newton 设置、传感器接触参数、卡扣弹簧与碰撞不变。该选择和原因写入采集配置，不改变其它任务。所有光学任务均分配 128 MiB 求解 arena，以容纳包括复杂 Peg 在内的源多片碰撞模型；不删除接触。
- Mac 仅替换 OpenGL 上下文；私有导入处理旧包路径、CPU checkpoint、NumPy 2 的 `mat` 兼容，不修改上游文件。当前引擎与作者旧 `mujoco_py` 的跨版本接触轨迹尚无逐帧等价证明。

## DIGIT / TACTO 光学选项

使用 [mujoco-tacto](https://github.com/utn-air/mujoco-tacto/tree/4c396eb802dcbd1f9e6647a31a011f37006c4fc1) `touch_playground_rot.xml` 的完整两网格传感器／长支架和四个接触垫。根部质量 15 g 与惯量直接来自 XML，其余 body 采用源编译值；不使用 standalone URDF 的占位惯量。胶面网格摩擦 `1 .005 .0001`，四垫摩擦 `2 .05 .0001`，均保留有效 source defaults。求解采用 MuJoCo 默认的 Newton／100（源 XML 未显式指定这两项），以及源示例的 implicitfast、elliptic cone、impratio=20、noslip=5、multiccd，容差来自示例 1e-8。

Renderer、`config_digit.yml` 默认平面光学胶面、内部相机／灯光、背景差分、平滑、随机 RGB 噪声和力相关光学位移均执行原代码；不再覆盖 `_generate_gel_trimesh`。240×320 背景为原 `bg_digit_240_320.jpg`。原力映射为 0–100 N 对应 0–5 mm 光学位移，不是夹爪限力；原噪声 N(0,7)，`--ideal` 关闭随机噪声。

必要的注册：由源接触垫前平面与配置中的 22 mm 光学距离建立刚性相机坐标系。源 `Sensor` 的全局 Euler 加角／坐标反号不适用于任意移动的双指，因此使用完整 SE(3) 变换和 `mj_contactForce` 的法向力聚合；相机只做刚性坐标注册，不换光学投影。多个胶面碰撞 geom 汇总到同一 sensor，同一物体多个接触点按力求和。允许任意接触物体，不设任务目标白名单。

每指 RGB、深度、相机深度、空载深度、有效区、接触区、力与时间戳分别输出。TACTO 原版不提供 FOTS 的 marker motion；界面不假造第三张流图。原函数独立进程对照、噪声统计、整体旋转、多接触、源机械参数、完整替换长指／保留开口、物理夹持和释放均有测试。

## DIGIT / FOTS 原生 MuJoCo 路径

固定 [FOTS-mujoco 分支](https://github.com/Rancho-zhao/FOTS/tree/4bd82cfc75ad8d600b2008df513b28cf04611610)，不使用主分支光学外接本地正交深度。完整导入 `digit_sensor.xml`：外壳、玻璃、胶体、曲面光学表层、摩擦层、30° 透视相机及参考 site；保留源 CAD、质量／编译惯量和接触参数。Panda 使用该来源默认 Newton／100／Euler、noslip=0，不借用 TACTO 或 AnySkin 的求解设置。源 solimp 超范围值由 MuJoCo 自身按引擎规则处理。

原 `env_test.py` 流程：内部相机 RGB 渲染后读取 legacy OpenGL 深度，左右翻转 `fliplr`；原 `ini_depth_extent.npy` 基线、`ini_bg_mlp.npy`、`digit_bg.npy`、`mlp_n2c_r.pth` 原样使用。内部裁剪面固定为来源 extent=1 对应的 0.001／1000 m，不受外部房间大小影响。`MLPRender.generate` 内部会修改基线数组，因此与作者每帧新建 renderer 的生命周期一致，每帧传基线副本，权重只加载一次。

FOTS 原平滑、法线／坐标 MLP、RGB、阴影与 `MarkerMotion` 全部执行。marker 使用各 sensor 参考 site 到接触物体 body 的相对位置／Rodrigues 旋转，保存初始与当前端点，**不对每帧位移重复积分**。原函数只用两个端点，故无需无限保存中间轨迹。失去接触、接触物体改变或 reset 时重建轨迹；无定时清零、流值限幅或局部锚点重置。保留源整数像素剪切、±50° twist 限制与 k=5 的箭头绘法，原始位移 `9×11×2` 不放大。

多物体的 RGB／深度可以渲染，但所接入 MuJoCo 示例的剪切／扭转计算只接受一条刚体相对位姿轨迹。若同时接触多个 body，`marker_status=multiple_bodies_unsupported`，marker 数组缺省并在 UI 提示，不用本地叠加近似替代。`no_contact` 时保留源基线残差产生的光学／膨胀结果，不自动补零。

输出区分 `render_input_depth`（无量纲非线性 OpenGL 缓冲）、`render_deformation`（原基线减缓冲）、`camera_depth_m` 与 `baseline_depth_m`（按裁剪面反投影的米制深度）、`indentation_m`（米制差）。源光学表层与摩擦层的间距会产生约 2 mm 图像深度，不能解释为已标定弹性压缩。实测本机空载约 4 µm 残差保留，显示比例不改数值。

源函数 RGB／阴影／流图逐像素对照、原场景深度读回对照、相机裁剪独立性、按压／平移／扭转／卸载、保持同状态重复渲染与 Panda 夹持均可重跑。源接触求解仍可能产生微小数值滑移；不能要求有真实相对运动时 marker 不变，也不会通过冻结输出使测试通过。

## DIGIT 双算法比较

`python -m wowskin_lab digit-compare --ideal` 对两套独立原模型施加相同探头指令，依次测试空载、按压、平移、扭转、倾斜、释放、再次接触和最终释放。球／方体／圆柱各 80 帧，10 Hz 采样；**物理状态和接触力不强制相等**。

每种探头保存 `frames.npz`、`metrics.csv`、`comparison.mp4` 与阶段 PNG。NPZ 的 `tacto/`、`fots/` 分别保存状态、指令、RGB、原始深度／基线、有效区／接触区、法向力、marker 状态和耗时；FOTS 另存 native 深度缓冲与标记流／数值位移。报告记录源码版本与各阶段结果。Panda `--demo` 单独保存真实接触夹起／抬升／释放过程，无 weld 附着。

TACTO 的 `depth_model=tacto_force_adjusted_optical` 与 FOTS 的 `fots_author_perspective_depth` 定义不同。深度或 RGB 的差值不能直接当作精度排名；需要配对真实 DIGIT 数据与相同实验条件才能评价真实性。该比较没有增益对齐、深度替换或同 CAD 假设。

## GelSight / 原生 MuJoCo 模型

`--sensor gelsight` 使用 [rlamprell/GelSightMujoco](https://github.com/rlamprell/GelSightMujoco) 的 `gelsight_mountable0.xml` 和配套外壳、后盖、玻璃 STL；来源版本、SHA256 和许可证状态见 [manifest](../assets/reference/gelsight_mujoco/manifest.json)。源码／资产通过脚本单独下载，不复制进本仓库。该旧模型不能改称 GelSight Mini，也不与 FOTS 的 GelSight 硬件标定混用。

原 body 内的前后盖、玻璃、半透明胶面、表层、隐形摩擦层和内部相机均保留。模块采用本地固定变换替换原 Panda 短指几何与惯量，长外壳轴沿手指方向，安装尾部朝手掌；源摩擦面的外边界作为安装参考，保留 80 mm 全开参考间距；这个安装尚未经过实物转接件验证，也不等同下述 Isaac Mini 装配。原相机视场角 20°、几何尺寸、质量和每层接触参数保留；只适配名称、固定安装姿态与碰撞 bit。来源光学表层不会与其目标物体直接碰撞，接触反力来自更靠里的摩擦／玻璃层，胶面图像由内部相机生成；这是来源的层结构，不是本地停止闭合或补零。

Panda 场景使用来源默认 Newton、100 次迭代、关闭额外 noslip；AnySkin 的 CG／30 次配置对这个硬接触模型不稳定，因此不沿用。夹爪仍为原位置伺服，没有新限力、最小开口、接触停闭或物体附着。原 `solimp` 中超出 MuJoCo 范围的值由引擎按照自身规则裁剪，并非有效弹性材料参数。

光学执行原 `gelsight_driver.py`：形变平滑、四路光照、Phong 漫反射／高光、内部阴影、背景、纹理噪声均保留。`RobotArm__BaseEnv.py` 中的工厂参数直接读取，默认分辨率 160×120；不导入旧 Gym 控制环境。原 `mujoco_py` 读取的是 RGB 渲染后的 OpenGL 深度缓冲。本地用原生 `mjr_readPixels` 重现；现代 Renderer 的普通深度模式会把透明层当作不透明，因此不能直接替代这一读取路径。

内部相机的近／远裁剪面固定到来源 PickAndPlace XML 编译出的米制值（约 1.481 mm／1481 m），同时保持原视场角。只改变独立的传感器渲染相机，不改变任务相机或物理参数。原光学算法直接使用非线性深度缓冲，因此继承新房间的 `extent`／裁剪面会错误地改变同一接触的光学响应；回归测试检查改变外部场景尺寸和裁剪设置后，原始深度与 RGB 保持一致。

**单位区分：** `render_input_depth` 是来源使用的无量纲 `[0,1]` 缓冲；`render_deformation` 是原光学函数平滑后的同单位结果；`render_depth_unit=legacy_opengl_buffer`。`baseline_depth_m` 从隔离空载的完整源模块渲染，不能以录制开始时的受压场景重新归零；`camera_depth_m`、`baseline_depth_m` 是根据真实近平面／远平面反投影后的米制深度；`indentation_m` 是两者之差，显示以 µm 标注。来源光学面与接触层相隔约 2 mm，因此图像深度可能接近 2 mm，即使刚性接触重叠很小；不能把它解读为已标定的硅胶弹性压缩。背景误差和浮点残差原样保存，不强行置零。

`--ideal` 只关闭原纹理随机噪声，保留原背景。该方法没有 FOTS 的标记剪切／扭转模型，所以不虚构标记流输出。测试覆盖原光学函数逐像素一致、噪声所在环节、透明胶面按压／释放、Panda 夹取／抬升／释放；抬升同时检查双侧接触和有限位移，避免把弹飞误判为抓取成功。

**复现证据范围。** 原 CAD 与原光学处理功能已经接入；源码对照证明相同输入下光学函数输出一致。还没有在作者旧 `mujoco_py` 引擎与当前原生 MuJoCo 上重放同一组接触并逐帧核对整条链路，因此不能把函数一致性称为跨版本传感器完全等价。安装姿态、碰撞位映射与引擎接口属于本地适配；实物力－压深／光学精度尚未验证。

## Isaac 项目的 GelSight 安装来源

核对了 TacEx 的 `adceed41afb7cb48f9ec1f66a662fb8e5a06627f` 和 UniVTAC 的 `0dafa10262e22f486f160a55d6f11aeab12d8e7b`：两者 Panda 装配引用的 `panda_arm_hand/instanceable_meshes.usd` 内容相同。保留 Panda 手掌及两根手指的开合关节，但 `panda_leftfinger/panda_rightfinger` 已使用**专用转接手指几何**，并非原装短手指保持不变。独立 GelSight Mini 外壳通过固定关节接到这些手指，胶面朝向夹持区。仅看 link 名没有变化，不能得出原 finger 未替换的结论。

- [TacEx 手指 URDF](https://github.com/DH-Ng/TacEx/blob/adceed41afb7cb48f9ec1f66a662fb8e5a06627f/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/panda_arm_hand/panda_arm_hand.urdf) 引用 `franka_double_gsmini_adapter` 几何；[装配 USD](https://github.com/DH-Ng/TacEx/blob/adceed41afb7cb48f9ec1f66a662fb8e5a06627f/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd) 用 `gsmini_left_joint/gsmini_right_joint` 固定传感器。下载并解析了 USD 和实际手指 STL；手指几何约 30×31.15×92.25 mm，与原 Panda 短手指不同。
- [UniVTAC 高分辨率腕部相机装配](https://github.com/univtac/UniVTAC/blob/0dafa10262e22f486f160a55d6f11aeab12d8e7b/third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd) 使用同一套基础手指、固定关节和 Mini 外壳；柔性胶体单独挂接。这不表示其胶体求解与刚性版本相同。
- [Mini 模型说明](https://github.com/DH-Ng/TacEx/blob/adceed41afb7cb48f9ec1f66a662fb8e5a06627f/source/tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/docs/README.md) 指明 CAD 来自 `gelsightinc/gsrobotics`，外壳 32×28×24 mm，胶面 25.25×20.75×4 mm。当前本项目旧 GelSightMujoco 外壳网格约 68×36×34 mm，是不同硬件，不能仅旋转或缩放成 Mini。
- [SoftVTBench 的机器人配置](https://github.com/TuojingAI/SoftVTBench/blob/58056111f01e05bf1a4ae1dee75db4e3d9e7c5be/SoftVTBench/source/tac_manip/tac_manip/assets/robots/franka.py) 指向 `Franka_gsmini/physx_rigid_gelpads.usd`；其外部资产、NeoSim 的实际资产安装尚未逐一核对，不据项目关系推断所有版本相同。

当前按用户允许的“独立模块装在 tip，或完整组件替换 finger”处理不同源模型；尚未移植 Mini 装配。上述旧 GelSightMujoco 是独立硬件选项，不能只换 Mini 外壳而沿用不对应的旧光学标定。

## MuJoCo 软体求解与光学传感器

MuJoCo [可变形物体文档](https://mujoco.readthedocs.io/en/stable/modeling.html#deformable-objects)提供 `flex/flexcomp`：网格接触、有限元材料模型以及三线性／二次降阶表示。本项目 AnySkin 的降阶指腹使用这条物理路线。`solref/solimp` 的软接触约束与胶体网格 FEM 是不同层次；刚体可有数值穿入，并不代表模拟了胶体内部应力。

TACTO、FOTS 和这里旧 GelSightMujoco 的光学变形／平滑不自动产生 FEM 材料反力。若要接软胶求解，应明确胶体网格、固定边界、材料参数，再把求解后的表面送给光学后端。当前保持各来源公开方法，不加入一套未经确认的材料模型来冒充作者实现。

## 玻璃与连续裂纹：后续研究方向

预分块模型可以接入现有机器人接触、碎片碰撞和录制流程，用于研究“碰撞后损坏／碎裂”的任务。但断裂只沿预定义界面发生；需显式定义界面强度、受拉／剪切判据、分块尺度及能量处理，不能把任意总接触力阈值称为实物玻璃强度。Houdini 的 [玻璃预破碎工具](https://www.sidefx.com/docs/houdini/destruction/glass.html) 和 [glue constraint 工作流](https://www.sidefx.com/docs/houdini/dyno/fracturing.html) 可供参考。

若研究目标是材料应力、裂纹起始和扩展，需专门断裂求解器。例如 [Glass Fracture Upon Ballistic Impact](https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2019.00239/full) 使用近场动力学模拟玻璃冲击碎裂；其实现基于开源 LAMMPS，[官方 peridynamics 文档](https://docs.lammps.org/Howto_peri.html) 提供断键与损伤模型。该论文的冲击载荷不等于机器人慢速夹持，接入与参数验证是另一项工作，不是 MuJoCo 中已有的开关。
