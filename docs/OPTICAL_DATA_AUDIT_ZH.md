# 光学触觉采集与 Isaac 对照检查

核对日期：2026-10-13。环境：macOS ARM64、Python 3.12.15、MuJoCo 3.15.0、NumPy 2.0.2。基于 `b19af84` 加本机工作区改动；这不是该提交本身的测试结论。

后续更新（2026-10-14）：已新增 `dataset-review` 和 `optical-act train|eval`，入口与小样本结果见[光学学习说明](OPTICAL_LEARNING_ZH.md)。以下来源对照与接口验收仍不代表策略任务成功率已达标。

范围是当前三种光学后端、Panda 任务接入、录制／选集／读回、可视化和学习入口，以及 TacEx／UniVTAC 的公开源码和一段实际数据。没有在本机运行 Isaac，也没有完成所有任务的整段示教成功率或策略训练验收。

## 结论与采集前缺口

当前可以分别录制 TACTO-DIGIT、FOTS-DIGIT、旧 GelSightMujoco 的双指图像与数值附件；原版 LeRobot 能读回五路视频。**这不等于三种光学传感器已经有可用的训练／部署闭环，也不等于与 Isaac GelSight Mini 数据一致。**

1. **已修复：光学 LeRobot 选集被错误拒绝。** `dataset_export.py` 原先对所有导出格式统一执行磁场要求，导致 `--format lerobot` 也报“需要 15 路磁场”。新增真实录制回归后，三种后端均复现此错误。现在只对 ViSk／FTF 拒绝；选一个成功 episode 重新导出 LeRobot，可以保留五路视频和无损光学 H5，源 metadata 不变。
2. **已补 ACT 光学适配，任务学习仍需验收。** 新增五路 RGB＋实测状态的 LeRobot ACT 基线、训练与 MuJoCo 实时执行。ViSk／FTF 原桥接器仍读取 15 路磁场，拒绝光学是正确行为。深度、marker、接触力不会仅因写入 H5 自动成为 ACT 输入。
3. **已补离线回放。** `dataset-review` 可把三相机、双指 RGB／压入深度／marker、100 Hz state/action 和评分按实际时钟浏览；支持原始 capture 与 LeRobot episode。在线 HUD、`digit-compare` 探头诊断和 `dataset-check` 完整性检查仍各自保留。
4. **不能混用 Isaac checkpoint。** 当前硬件、光学标定、marker、图像尺寸、动作标签均有差异。不能改键名、缩放图片后就声称兼容 UniVTAC。
5. **来源记录不足以防止跨版本混录。** 光学 H5 保存来源 manifest；追加签名比较配置，但尚未包括实际 MuJoCo 版本、运行代码和所有资产摘要。升级引擎或代码后仍应使用新目录。该缺口也适用于磁场采集。
6. **任务验收仍需完整操作。** 本轮补入当前双端长线绕柱，三种后端各覆盖方块、单卡扣线、绕线、喂食、Peg、薯片的初始化／短时步进／真实录制，共 18 个组合。它们不证明绕线成功、Peg 插入成功或薯片无损抓取成功。已有独立方块抓取／抬升／释放测试另行通过。

“多传感器”当前指分别启动不同配置采集。一次运行选择一种硬件／算法，两个光学手指同型号；没有同一 episode 混装 AnySkin＋DIGIT 或左右不同型号的统一 schema。数据应按任务、传感器、算法与配置分目录。

## 当前代码到底迁移自哪里

- **TACTO-DIGIT**：`utn-air/mujoco-tacto`，固定 `4c396eb802dcbd1f9e6647a31a011f37006c4fc1`。原 Renderer、光学配置、背景与力相关光学位移来自该 MuJoCo 移植；本地改接刚体坐标、接触力和 Panda 安装。场景可视指部／支架的 alpha 本地设为 1，覆盖源示例的 0.2，长指网格 `finger_0` 本地改为黑色，供场景 RGB 正常遮挡；触觉渲染与碰撞参数保持原来源。
- **FOTS-DIGIT**：`Rancho-zhao/FOTS` 的 `FOTS-mujoco` 分支，固定 `4bd82cfc75ad8d600b2008df513b28cf04611610`。使用 DIGIT 模型、深度缓冲基线、法线／坐标到颜色的 MLP 与 MarkerMotion。
- **旧 GelSight**：`rlamprell/GelSightMujoco`，固定 `a060b0ec384b189e39d9576df753d381857bcc56`。使用该硬件外壳／层结构和 Phong 光学函数；它不是 GelSight Mini。

本轮按 [TACTO](../assets/reference/tacto/manifest.json)、[FOTS](../assets/reference/fots/manifest.json)、[GelSight](../assets/reference/gelsight_mujoco/manifest.json) 检查了共 **33 个文件 SHA256，全部一致**。这里只验证 manifest 列出的文件，不能扩大成整个第三方仓库都已逐文件验证。

因此，“当前光学使用刚体接触＋光学生成”主要来自这些来源模型的设计；Panda 安装、坐标映射、现代 MuJoCo 接口和录制属于本地适配。它们没有移植 TacEx／UniVTAC 的 UIPC 胶体求解。刚体软接触约束允许数值穿入，光学平滑／位移可以生成凹痕图，但不等同于胶体内部 FEM 应力求解。

## TacEx 与 UniVTAC 不是同一个固定配置

检查版本：TacEx `adceed41afb7cb48f9ec1f66a662fb8e5a06627f`；UniVTAC `0dafa10262e22f486f160a55d6f11aeab12d8e7b`。

TacEx 默认 GelSight Mini 的 RGB 来自 **Taxim**，marker 可配置为 **FOTS**；框架还提供 UIPC 和基于软体表面的 marker 路径。不要把“TacEx 使用 FOTS marker”理解为“TacEx RGB 就是本地 FOTS 的 DIGIT MLP”。来源：[TacEx 项目](https://github.com/DH-Ng/TacEx)、[固定 Mini 配置](https://github.com/DH-Ng/TacEx/blob/adceed41afb7cb48f9ec1f66a662fb8e5a06627f/source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py)。

UniVTAC 的 [任务传感器工厂](https://github.com/univtac/UniVTAC/blob/0dafa10262e22f486f160a55d6f11aeab12d8e7b/envs/sensors/tactile.py) 又覆盖了默认配置，采用 UIPC 胶体和 ManiSkill marker 路径。其源码默认 marker 参数不能反推历史公开 episode 的点数；必须检查数据本身。下面下载的文件实际是 1,200 个数值点，预览中可见黑点的数量也不能代替数值数组维度。

## 实际下载的数据与图像

从 [UniVTAC 官方数据集](https://huggingface.co/datasets/byml/UniVTAC) 下载：

- 数据 revision：`e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd`。
- 文件：`isaac45/lift_can/hdf5/0.hdf5`，132,373,510 字节，170 帧。
- SHA256：`455575219f986a9a8ddc05739110735ada09104f38923944bd794a61bb0ed947`。
- 取第 0、106、169 帧检查；106 是本段左指相机深度最小的帧，不是未经验证的“最大力”标签。

本机证据保存在 `outputs/diagnostics/optical_audit_20261013/`，包括下载来源、原 HDF5、字段清单、源码摘取、本地新跑的三套按压／释放和图像。该目录被 Git 忽略，不随仓库分发。

- [四种输出并排图（本机）](../outputs/diagnostics/optical_audit_20261013/images/optical_comparison.png)
- [Isaac 原段的首帧／接触／末帧（本机）](../outputs/diagnostics/optical_audit_20261013/images/isaac45_samples.png)
- [样本与数值摘要（本机）](../outputs/diagnostics/optical_audit_20261013/sample_inspection.json)

本地是球探头按压，Isaac 是罐体抓取；硬件、姿态、载荷也不同。并排图用于检查输出类型、原生尺寸、背景和 marker 表达，**不能用于 PSNR／SSIM 精度排名或跨引擎物理等价结论**。触觉 RGB 保持原生尺寸与像素值；只有外部相机预览及单独的箭头预览缩放。

### 字段差异

- **RGB**：本地 DIGIT 为 H×W=`320×240`；旧 GelSight 为 `120×160`；公开 Mini 样本为 `240×320`。不是同一硬件仅换图片大小。样本有双指 `rgb` 和叠黑点的 `rgb_marker`；本地 FOTS 的绿色箭头图是另一个可视化字段，不能拿它冒充黑点触觉图。
- **存储与颜色**：本地光学 H5 保存无损 uint8 RGB，同时写入 LeRobot 编码视频。Isaac 样本的 `rgb`／`rgb_marker` 是 HDF5 内 JPEG 字节。本轮按作者 `HDF5Handler.stream_to_img` 的 OpenCV 解码路径读取，没有额外交换通道。迁移时应逐一核对生成端和解码端，不能只见 `rgb` 名称就统一增加 BGR→RGB。来源：[数据写读代码](https://github.com/univtac/UniVTAC/blob/0dafa10262e22f486f160a55d6f11aeab12d8e7b/envs/utils/data.py)。
- **深度**：Isaac 样本的 `depth` 来自 `height_map`，是毫米制相机距离，左指范围约 27.741–34 mm，34 mm 是远裁剪背景；不是压痕深度。TacEx 的 `camera_depth` 还可能是归一化 uint8 调试图。本地 `camera_depth_m`／`baseline_depth_m`／`indentation_m` 用米，FOTS／旧 GelSight 另保留无量纲 OpenGL 输入缓冲。不能直接相减或归一化成同一物理量。来源：[传感器输出映射](https://github.com/univtac/UniVTAC/blob/0dafa10262e22f486f160a55d6f11aeab12d8e7b/envs/sensors/tactile.py)、[TacEx 深度实现](https://github.com/DH-Ng/TacEx/blob/adceed41afb7cb48f9ec1f66a662fb8e5a06627f/source/tacex/tacex/gelsight_sensor.py)。
- **marker**：公开文件每侧为 `(170,2,1200,2)`，两组像素端点；本地 FOTS 每帧为 `(9,11,2)` 位移。TACTO／旧 GelSight 没有该字段；FOTS 多 body 接触会明确报告不支持，不能补零当成有效样本。
- **动作与状态**：该 Isaac 文件保存 `(170,9)` 的关节反馈，未保存本项目的实际下发 TCP 目标。作者 ACT 预处理用 `q[t]`／`q[t+1]` 构造 state／action。本地 LeRobot 是七维绝对 TCP xyz＋rotvec＋夹爪目标，并分开保存实测 state 和命令；不能只改成 UniVTAC 字段名。来源：[数据转换](https://github.com/univtac/UniVTAC/blob/0dafa10262e22f486f160a55d6f11aeab12d8e7b/envs/utils/data.py)、[策略预处理](https://github.com/univtac/UniVTAC/blob/0dafa10262e22f486f160a55d6f11aeab12d8e7b/policy/_base_data_preprocessor.py)。
- **时钟**：公开样本保存 `step=112,114,…,450`，没有独立逐帧秒时间戳，不能仅凭视频播放速度推定训练采样周期。本地有实际仿真时间和名义 30 Hz 视频索引，原始机器人数据为 100 Hz。

UniVTAC 官方还明确区分 Isaac 4.5／5.1 数据及 checkpoint；本轮实际解码的是 **4.5 的一个 episode**，没有下载并验收 5.1，也没有证明单个样本代表全部公开任务。见其[数据版本说明](https://github.com/univtac/UniVTAC/blob/0dafa10262e22f486f160a55d6f11aeab12d8e7b/README.md)。

## 本机检查结果

- 初始光学组合测试：22 项通过，无跳过。涵盖来源机械参数、独立源函数图像对照、相机深度读回、噪声、按压／卸载、旋转、多接触、Panda 抓取、任务接入和 LeRobot 录制读回。
- 选集导出新增回归：修改前 3 个后端均失败；修复后全部通过。检查五路视频、无损 RGB／深度／时间戳、原 metadata 不变，并继续拒绝 ViSk／FTF 光学导出。
- 任务组合扩充：18 个组合均通过，包含当前双端长线绕柱。测试仅短时保持与录制；不是 18 个任务成功示范。
- 通用数据回归：LeRobot 录制、公共学习导出与光学时间戳三个模块共 9 项通过（其中时间戳 2 项也计入上面的 22 项）。静态检查、Python 编译和 `git diff --check` 通过。
- 三套独立球探头按压／释放重新运行，输出均有限且有接触响应。释放后 TACTO／旧 GelSight 回到各自零压痕基线；FOTS 保留来源约 4 µm 的空载残差。没有改背景、统一深度增益或替换物理模型来对齐图像。

复跑入口：

```bash
.venv/bin/python -m unittest \
  tests.unit.test_optical_timestamps \
  tests.physics.test_digit_sources tests.physics.test_optical_backends \
  tests.physics.test_fots tests.physics.test_optical_tactile \
  tests.physics.test_optical_tasks tests.official.test_optical_recording

# 成功光学数据可选集导出；输出目录必须尚不存在
.venv/bin/python -m wowskin_lab dataset-export data/episodes_tacto/pick_cube \
  --format lerobot --episodes capture_000000 --output data/exports/tacto_cube_subset
.venv/bin/python -m wowskin_lab dataset-check data/exports/tacto_cube_subset
```

上述源函数一致性证明的是“相同输入下调用原算法的结果”，不是旧 MuJoCo／新 MuJoCo／Isaac 的整条接触链路等价。完整训练、硬件标定、所有任务成功率仍未验证；Robotiq＋AnySkin 已知翻转问题也未在本轮修复。

## 其它 MuJoCo 方案

除了已接入的三个来源，核对到以下公开实现：

- [L3S/TACTO-MuJoCo](https://github.com/L3S/TACTO-MuJoCo)：另一条 TACTO 的 MuJoCo 集成，MIT；文档使用 Python 3.8／旧 dm-control，并明确不提供精确的接触形变／摩擦动力学。本轮未安装运行，不能视为更准确或直接兼容 MuJoCo 3.15 的替换件。
- [Rancho-zhao/Digit_PhongSim](https://github.com/Rancho-zhao/Digit_PhongSim)：DIGIT 的 MuJoCo＋Phong 示例，来源测试环境为 MuJoCo 2.2.2。与现有路径同属光学生成方案，不能据此宣称解决软胶 FEM 或 Isaac Mini 对齐；本轮未运行。
- [MuJoCo 官方 touch_grid](https://github.com/google-deepmind/mujoco/blob/3.15.0/plugin/sensor/README.md) 和 [AIST MuJoCo 触觉插件](https://github.com/isri-aist/MujocoTactileSensorPlugin)：接触力／力矩或法向力阵列，适合另一类触觉观测；不会直接生成 GelSight RGB，不是光学传感器替代品。
- [Taxim 原版](https://github.com/Robo-Touch/Taxim) 能接收深度图，可作为未来跨引擎共用的光学层；它本身不是本项目已经接好的 MuJoCo Mini 力学模型。若要对齐 TacEx，应明确移植同一 Mini 硬件、标定、相机和深度定义，再决定胶体求解方式。

独立 ACT 数据读取／训练／执行契约和统一离线回放已补齐，后续仍需每个任务的小批完整示范与策略成功率验收。若研究目标要求跨 Isaac／MuJoCo 共用触觉编码器，还需同一 Mini 模型、相同探头／姿态／载荷轨迹的配对对照；目前这些证据尚不存在。
