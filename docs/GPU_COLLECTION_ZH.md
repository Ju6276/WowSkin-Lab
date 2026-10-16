# Mac / NVIDIA GPU 共用采集入口

同一份仓库，Mac 默认原生 CPU MuJoCo，Linux NVIDIA 可显式选择 `--physics-backend warp`。固定版本：Python 3.12、MuJoCo / MuJoCo Warp 3.15.0、Warp 1.18.0。GPU 路径已接入环境 `step`、遥操作、自动采集和原始记录器；薯片动态断裂暂明确拒绝 GPU，继续使用 CPU；任务是否完成、是否实时要分别验收。当前仍是可选实验后端，**不能整体替代 CPU 默认实现**；具体结果见 [4060 验收记录](GPU_VALIDATION_ZH.md)。GPU 后端不会自动修复 Robotiq 软指腹强闭合翻转。

## 安装

在 NVIDIA 机器克隆项目，不需要另建一个项目或复制旧实验工作区：

```bash
git clone https://github.com/Ju6276/WowSkin-Lab.git
cd WowSkin-Lab
python3.12 scripts/setup_capture.py --gpu
.venv/bin/python scripts/setup_upstream.py --only franka tacto fots_mujoco gelsight_mujoco
.venv/bin/python -m pip install -r requirements/tacto.txt
.venv/bin/python -m pip install --no-deps -e third_party/mujoco-tacto
```

Mac 使用原 README 的安装命令，不加 `--gpu`，无需 Warp。无显示器的 NVIDIA 机器可设置 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` 运行自动采集；遥操作窗口需要 Linux 桌面显示，不能用无窗口的 EGL 会话代替人工验收。

## 物理执行与兼容选项

- GPU：动力学、碰撞／约束、原 trilinear 指腹、迁移的原绳子弹性力；默认 IK/PD 控制的 DLS 逆运动学与关节力矩；AnySkin 磁场／极分解、体积检查、接触力与滑移；方块、喂食、single-clip、routing、Peg 的几何评分和成功保持状态。使用 CUDA Graph，普通控制周期将连续物理步、末态观测和评分合在一张 Graph 中。
- CPU：初始化／复位、操作输入、界面、文件编码／保存；传感器小向量电子响应及来源光学图像的部分预后处理。Pollen／ReSkin 网络和 FOTS MLP 在选择 Warp 后端时运行在 CUDA，仍沿用原权重和算法。FOTS 跨设备浮点舍入可能产生 1 个灰度级的差异，同设备来源对照仍要求逐像素一致。
- 普通 IK/PD 任务每个传感器周期上传目标，返回供现有观察／显示／记录接口使用的紧凑快照；不再完整导出质量矩阵、约束 Jacobian 等求解工作区。喂食头部力、峰值、冲量仍在 GPU 每个物理步计算，不降低峰值采样频率；限时扰动也在 GPU 执行。GUI 使用已求解的脱离运行状态的快照，不再为显示重复 CPU `mj_forward`。
- 可选 Cartesian／旧控制模式、接触停止或自定义逐物理步回调仍采用 CPU 反馈兼容路径，不能将这些模式称为完整 GPU 控制。默认采集的 IK/PD 路径不走该分支。
- 场景相机默认 `--camera-backend opengl`，保持原 MuJoCo 图像。显式选择 `--camera-backend warp` 后，三路场景 RGB-D 由官方 MuJoCo Warp 渲染器直接读取 GPU 物理状态，使用渲染 Graph；录制与遥操作相机预览共用该选项。它的光线追踪颜色、阴影、深度离散结果与 OpenGL 不逐像素相同，数据元信息记录渲染器，必须使用独立采集目录。双指 TACTO／FOTS／GelSight 仍走各来源成像流程，不能用场景渲染器冒充。
- 原 CPU 绳子插件没有被 MuJoCo Warp 原生支持。本地 `core/gpu/cable.py` 将 3.15.0 插件的本构关系、截面刚度、参考曲率和力矩变换翻译为 Warp 被动力内核。链节、质量、几何和材料参数保持来源定义；没有 MPM。应力着色未迁移。

迁移这些计算不保证单环境实时。RTX 4060 上仍须分别测量控制周期、相机生成、编码／写盘；GPU 内部求解也可能超过 10 ms 的 100 Hz 控制预算。具体测量见验收记录。

以下变化均需显式选择，并写入每条记录的 `simulation_config.physics_execution`：

- `--gpu-disable-multiccd`：关闭多点 CCD，保留几何与接触 margin。
- `--gpu-disable-nativeccd`：采用替代碰撞路径，使原非零网格接触 margin 可以导入。接触流形可能与 CPU 默认算法不同。
- `--gpu-disable-noslip`：关闭 Warp 不支持的额外 no-slip 求解阶段，摩擦系数保留。可能影响滑移和抓持，不能宣称与原默认接触解完全等价。
- `--gpu-solver-iterations N`、`--gpu-solver-tolerance X`：显式改求解上限／容差，默认保持来源值。提高上限不保证收敛。
- 自由度超过 60 时使用 Warp 所需的稀疏 Jacobian，自动记录这一变化。

浮点精度为 GPU float32，CPU float64；碰撞轨迹不承诺逐位相同。状态采样时间按实际执行的物理步计数形成双精度时间戳，不靠复制帧补齐。记录的姿态和接触力直接从 GPU 下载；常规采集期间不会调用 CPU `mj_forward` 重算一份力来替换 GPU 结果。

GPU 溢出、迭代达到上限或非有限状态会中止录制并要求复位，不回退到 CPU。中断数据保留原因和诊断，不作为成功数据自动导出。达到迭代上限表示未满足当前停止条件，本身不等于网格已翻转。

## 采集和检查

使用 GPU 物理和直接 GPU 场景相机录制抓方块：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m wowskin_lab record-auto \
  --task pick_cube --sensor-backend tacto --physics-backend warp --camera-backend warp \
  --gpu-disable-multiccd --gpu-disable-nativeccd --gpu-disable-noslip \
  --ideal --episodes 1 --output data/gpu_warp_camera_cube
```

保留原场景图像则使用 `--camera-backend opengl`（默认）；它也能与 GPU 物理一起使用。CPU 后端不支持 Warp 相机，入口会明确报错。

TACTO / Panda 弹性 single-clip 的完整试录入口（最终代码重跑未达任务成功，当前用于验收，暂不建议批量采集）：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m wowskin_lab record-auto \
  --task cable --cable-layout single_clip --clip-model elastic_beam \
  --sensor-backend tacto --physics-backend warp \
  --gpu-disable-multiccd --gpu-disable-nativeccd --gpu-disable-noslip \
  --ideal --episodes 1 --output data/gpu_tacto_elastic

# 仅成功并打包后才有 lerobot/；失败数据只保留 raw/ 和任务评分
.venv/bin/python -m wowskin_lab dataset-check data/gpu_tacto_elastic/cable_insert/lerobot
.venv/bin/python -m wowskin_lab dataset-review data/gpu_tacto_elastic/cable_insert/lerobot \
  --output outputs/gpu_tacto_elastic_review
```

这是读取仿真物体位姿的规则控制器，非 agent 推理或人工示范。成功仍由现有任务评分判定；single-clip 分数是线中心进入卡扣并保持，不等于松手后弹性锁扣保持测试。

Linux 桌面遥操作使用同一套 backend 参数，例如：

```bash
.venv/bin/python -m wowskin_lab teleop --task cable --cable-layout single_clip \
  --sensor digit --optics tacto --clip-model elastic_beam --input keyboard \
  --physics-backend warp --gpu-disable-multiccd --gpu-disable-nativeccd --gpu-disable-noslip \
  --record-dir data/gpu_keyboard --ideal
```

Mac 仍用 `.venv/bin/mjpython`，不传 GPU 参数。不同物理后端／兼容设置使用独立数据目录。大数据、缓存和回放均不提交 Git。

## 回归测试

```bash
.venv/bin/python -m tests unit
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m tests gpu
.venv/bin/python -m tests physics
```

部分原 CPU 物理检查使用 Pollen 原模型，先执行 `.venv/bin/python -m wowskin_lab pollen download`。ViSk／FTF 数据链路测试还需按学习文档获取固定版上游（`scripts/setup_upstream.py --only visk ftf`）。

`gpu` 真正在 CUDA 上执行：原插件与 GPU 力／轨迹对照、原生一步与 Graph 对照、环境复位和故障保护、各传感器 RGB-D 写入与回放、任务短轨迹对照，以及 GPU IK／磁场／接触读数／评分与原 CPU 公式的同状态核对、直接 GPU 场景相机。任务对照给 CPU 和 GPU 相同的兼容设置，以区分执行误差与接触算法变化。它不替代完整任务成功评估。没有 CUDA／可选包时 GPU 组明确跳过，不能把跳过算为 GPU 验收。

原 `physics` 组检查默认 CPU 行为，即使在 NVIDIA 机器上运行也不会自动变为 GPU 测试。Robotiq 强夹线仍可能在此组触发已知翻转错误，须在报告中列明。
