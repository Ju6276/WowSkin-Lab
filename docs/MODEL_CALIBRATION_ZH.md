# 传感器模型与标定

本项目的 WowSkin 是本地物理近似模型，尚未完成实物材料与磁场标定。启动与录制见 [README](../README.md) 和 [操作说明](PICK_CUBE_ZH.md)。

## 计算方式来自哪里

AnySkin 的磁化皮肤受接触后变形，改变下方五个三轴磁力计的读数。[AnySkin 论文](https://arxiv.org/abs/2409.08276)

本项目用 MuJoCo 弹性四面体网格近似皮肤形变，再选择磁场输出方式；它不是 AnySkin、ViSk 或 FTF 官方提供的传感器仿真。三个方式共享同一套接触力学：

- **原偶极子（默认）**：每个四面体中心放置磁偶极子，位置随网格移动，方向随形变梯度的旋转部分改变；叠加五芯片处磁场并扣除基线。
- **拟合偶极子**：追加 `--magnetic-fit configs/magnetic_fit_pollen_nonslip.json`，在偶极子场上应用五个正幅值系数。运行时不查询网络。
- **网络预览**：追加 `--reskin-model outputs/reskin_forward_mapped --reskin-extrapolate`。从接触提取力加权 XY 和最大顶部压缩深度，输入本地 `3→64→64→15` 网络；没有输入完整网格、剪切、接触面积或物体形状。

拟合与网络选项互斥；`--ideal` 关闭合成噪声和漂移，不关闭接触形变。偶极子路径保留滤波与延迟；网络直接输出，不应把两者的时序差异误认成物理精度。

### 磁场公式

体元初始磁矩为 `m_j = M V_j e_z`，随材料旋转但大小固定。每个芯片叠加：

`B_i = μ0/(4π) Σ_j [3 r̂_ij (m_j·r̂_ij) − m_j] / |r_ij|³`

芯片位置减去体元中心得到 `r_ij`，长度用米；T 转成 μT 后输出基线差值 `ΔB`。实现：[physics.py](../wowskin_lab/core/physics.py)。均匀磁化、点偶极子离散和材料参数均为模型选择，未验证真实磁滞或网格收敛。

## 当前仿真幅值拟合

运行配置：[magnetic_fit_pollen_nonslip.json](../configs/magnetic_fit_pollen_nonslip.json)。按 PCB **U1中心/U2上/U3右/U4下/U5左** 顺序，倍率为 **54.525 / 15.870 / 50.256 / 65.217 / 27.875**。

拟合参考是 `Pollen 不滑移磁场 → 本地 MagNet 架构逆模型的 XYZ 伪标签 → 已训练前向网络 → 五芯片模长`。使用 cable.csv 中筛选后的 1691 帧，以及仿真手动夹持记录中的 1986 帧；Pollen 标签 0 不保证夹紧，因此还用了最大芯片模长 ≥50 μT 的筛选规则。Pollen 不提供位置、深度或力真值。

两组数据并非同一接触实验，因此拟合的是各芯片模长的 10/25/50/75/90 百分位。每个芯片三个轴乘同一正数，保留方向、符号和零点；基线一起变换，噪声在缩放后单独添加。五个读出倍率不代表已辨识的材料磁化强度，且不会增大夹爪力。

按完整片段/时间分割训练、验证、测试；测试段对网络参考的模长分位数 RMSE 为 257.52 μT，对 Pollen 实测模长分布为 192.87 μT。**这是幅值分布误差，不是逐帧 15 路磁场误差。** 此配置仅在线材上拟合，勺子/方块的迁移与真机精度未验证。

[拟合程序](../scripts/fit_pollen_dipole_amplitude.py) · [来源、划分和报告](../outputs/diagnostics/pollen_nonslip_dipole_fit/report.json)。重跑时指定新的 `--output`，不覆盖已有结果。

## 参数来源

- 外形：OnRobot 参考 STL 按厂商平面图缩放至约 42.2×43.6 mm；底座约 11.5 mm、套硅胶后总高约 13.5 mm 来自用户测量。顶部增厚取差值 2 mm。用户确认只包上沿，具体侧壁厚度和下包长度尚未测量。
- 材料：杨氏模量 30 kPa、泊松比 0.35、密度 1000 kg/m³、摩擦系数 0.6；弹性阻尼 1e-5、活动节点阻尼 0.002。这些值尚未在该设备上辨识。
- 磁场：均匀 +Z 磁化 4000 A/m、芯片至软层底面间距 2.2 mm，均未实测辨识。
- 信号：100 Hz；独立高斯噪声标准差 0.5 μT、一阶滤波时间常数 10 ms、延迟 1 个采样。噪声不累加，线性漂移默认关闭。
- 标签：接触阈值 0.005 N、滑移速度阈值 0.0005 m/s，为本地标签规则，不是 Pollen 的人工标注。
- 数值：线材/喂食软层启用内部接触，数值接触半径 0.2 mm、时间常数 0.2 ms、体积比中止阈值 0.15。线材软层的接触阻抗 `solimp` 为 `0.98 0.98 0.001 0.5 2`，限制薄边缘网格的过度穿透；它影响内部和外部接触的顺应性，不是实测材料参数。积分步长与材料物性是不同参数，当前尚无完整收敛验证。[MuJoCo 内部接触说明](https://mujoco.readthedocs.io/en/3.5.0/XMLreference.html#flex-contact)

配置：[sensor.json](../configs/sensor.json)、[sensor_hardware.json](../configs/sensor_hardware.json)。

## 夹爪配置来源

[gripper_control.json](../configs/gripper_control.json) 使用固定版本 Google DeepMind Menagerie Panda 位置执行器，来源与哈希在 [manifest](../assets/reference/panda_menagerie/manifest.json)。原位置增益 100 N/m、速度增益 10 N·s/m、tendon 限幅 ±100 N，两指传动各乘 0.5。

本地关节坐标 `q_local=q_source−0.04`，闭合指令方向相反；实际 tendon 输出为 `−4×closure−100×q_local−10×qdot`。因此 ±100 N 不是恒定夹紧力。新增软指腹的接触负载仍可能顶开位置伺服；不能用磁场倍率解决该问题。

线材默认使用固定版本 [robosuite Panda 夹爪参数](../assets/reference/robosuite_panda/manifest.json)，每指独立位置执行器，刚度 1000 N/m、限幅 ±20 N，关节 damping=100、armature=1、frictionloss=1。保留本地软指腹安装和行程；每指目标为新增指腹行程下限乘以闭合比例，空夹时保持 2 mm 间隙；不把源模型的 40 mm 行程继续压到本地限位之外。这些是 benchmark 仿真设置，未对 WowSkin 真机夹持力标定。

机械臂本地 PD 控制和 IK 不等于完整官方 Franka 控制器。`--grip-force` 显式启用的旧限力模式没有官方标定依据，不作为默认配置。

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

共用底座/硅胶套尺寸；任务网格与独立传感器网格分辨率不同，未验证网格收敛。皮肤受力、接触法向力和磁场分别保存，不能把五个磁场点视为五个测力单元。

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
