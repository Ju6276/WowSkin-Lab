# 代码与目录

项目：项目根目录。旧 `WowSkin-Sim` 路径不再使用。

```text
wowskin_lab/
  core/       软体接触、几何和磁场
  envs/       传感器、Franka、方块、线材、喂食、鞭子
  teleop/     七维动作、Pico、SpaceMouse
  ui/         窗口输入与触觉图
  data/       ViSk / FTF 原始采集与格式转换
  learning/   官方源码、配置准备和核验
  baselines/  Pollen、ReSkin 前向模型、MagNet 逆模型
  legacy/     已停用的简化实验
configs/      传感器、夹爪、幅值拟合、固定上游版本
assets/       原素材、CAD、许可证及来源 manifest
third_party/  上游源码和依赖；official/ 为固定版本参考
scripts/      下载、安装、模型分析及检查入口
tests/        分组检查：unit / physics / official / artifacts / legacy；manual 为窗口手测
data/         示教和导出数据
outputs/      模型、标定原始记录、当前采集与必要分析
docs/         操作、学习、标定、目录；baselines/ 为 Pollen
```

## 数据与模型

- `outputs/reskin_forward_mapped/`：当前候选顺序的前向网络。
- `outputs/reskin_inverse_mapped/`：本地 MagNet 架构逆模型。
- `configs/magnetic_fit_pollen_nonslip.json`：运行用五芯片幅值系数。
- `outputs/calibration/`：真实传感器记录。
- `outputs/cable/`、`cable_insert/`、`feeding/`、`pick_cube/`：手动会话。
- `outputs/official/`：官方学习流程生成的工作副本与点标注 notebook。
- `outputs/diagnostics/`：模型评估、拟合依据、通道检查与保留的学习结果。

旧试跑输出和文档在项目外的 `~/Library/Application Support/WowSkin-Lab/archives/`。启动不依赖该目录。

## 环境

新机器先按根目录 [README](../README.md#安装mac) 创建 `.venv`。需要官方训练依赖时：

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-acquisition.txt -r requirements-learning.txt
.venv/bin/python scripts/setup_upstream.py
```

点标注另装 `requirements-annotation.txt`；官方 CUDA 环境以各上游环境文件为准。GUI 使用 `mjpython`，无窗口处理使用 `python`。

## 动作与记录接口

`step([dx,dy,dz,drx,dry,drz,grip])`：世界 XYZ 位移（米）、末端局部旋转向量（弧度）、夹爪绝对闭合目标 0–1。旋转按 `R_target @ Exp(dr)` 组合。`env.hold_action()` 保持现有夹爪目标；全零动作会打开夹爪。

`ViSkSimRecorder`、`FTFSimRecorder` 分别保存两种原始格式，`LearningSimRecorder` 同时写入两套。真实 Franka 后端尚未接入。

`third_party/official/` 保持固定原版。ViSk 论文单传感器配置只在生成的 `outputs/official/` 工作副本中应用，补丁和 manifest 随副本保存。官方训练入口见 [学习工作流](LEARNING_WORKFLOWS_ZH.md)。

## 检查

```bash
.venv/bin/python -m wowskin_lab official status
.venv/bin/python -m pip check
.venv/bin/python -m tests unit       # 无设备、无训练权重的快速检查
.venv/bin/python -m tests physics    # MuJoCo 与夹持，耗时较长
.venv/bin/python -m tests official   # 需要下载官方源码和采集依赖
.venv/bin/python -m tests artifacts  # 需要本地公开数据、训练结果
.venv/bin/python -m tests legacy     # 已停用实验的兼容性
.venv/bin/python -m tests all --list # 列出全部检查，不执行
```

全量物理测试耗时较长；已知夹持问题见 [操作说明](PICK_CUBE_ZH.md#使用范围)。仓库：[Ju6276/WowSkin-Lab](https://github.com/Ju6276/WowSkin-Lab)。`tests/manual/` 只用于交互窗口检查，不参与自动测试。旧格式导出检查放在 `legacy`，验证保留数据的读取，不代表官方训练流程。

`artifacts` 检查已有训练结果；缺少可选结果时跳过。`physics` 的夹持测试可能暴露当前已知问题，不能把快速检查通过解释为物理任务已验证。下载的依赖、权重、用户采集和生成输出保留在本机，由 `.gitignore` 排除。
