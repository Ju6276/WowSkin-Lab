# WowSkin-Lab

Franka 的 MuJoCo 遥操作与触觉采集：左指 WowSkin、右指普通硅胶，支持键盘、SpaceMouse 和 Pico。采集与训练流程面向 ViSk、FTF；真实 Franka 后端尚未接入。

## 安装（Mac）

```bash
git clone git@github.com:Ju6276/WowSkin-Lab.git
cd WowSkin-Lab
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements-acquisition.txt
.venv/bin/python scripts/setup_upstream.py --only franka
```

已安装的本机环境可直接启动。官方算法源码另用 `scripts/setup_upstream.py --only visk ftf` 下载，训练依赖和数据处理见[学习流程](docs/LEARNING_WORKFLOWS_ZH.md)。采集数据、训练权重、虚拟环境和下载的上游仓库不纳入 Git。

## 启动场景

在项目目录运行；Mac 的 MuJoCo 窗口使用 `mjpython`。

```bash
# 简单插线：固定线材一端，单卡槽
.venv/bin/mjpython -m wowskin_lab teleop --task cable --cable-layout single_clip --input spacemouse --ideal

# 绕线 / 方块 / 喂食：选一个运行
.venv/bin/mjpython -m wowskin_lab teleop --task cable --input spacemouse --ideal
.venv/bin/mjpython -m wowskin_lab teleop --task pick_cube --input spacemouse --record-format both-raw --ideal
.venv/bin/mjpython -m wowskin_lab teleop --task feeding --input spacemouse --ideal
```

键盘测试将 `--input spacemouse` 改为 `--input keyboard`；Pico 改为 `--input pico`，连接设置见[操作说明](docs/PICK_CUBE_ZH.md)。喂食从夹住装有豆子的勺子开始。

默认使用 **MuJoCo 形变＋磁偶极子**，不是网络预测。需要之前的 Pollen 幅值拟合预览时，追加 `--magnetic-fit configs/magnetic_fit_pollen_nonslip.json`；它不是实物物理标定，方块和勺子的迁移尚未验证。

## 操作与录制

先点击窗口。SpaceMouse 推拉/倾斜控制六轴，左键夹紧、右键张开、双键保持开度。

- 键盘：W/S 前后，A/D 左右，E/Q 上下；I/K、L/J、U/O 旋转；X/Z 夹紧/张开。
- **B** 复位并开始录制，先保持无接触约 1 秒仿真时间。
- **Enter / N** 保存成功 / 失败；**R** 复位；**V** 切视角；**P** 保存中断记录并退出。

双格式原始记录保存在 `data/official/`；使用幅值拟合预览时保存在 `outputs/diagnostics/fitted_dipole_preview/episodes/`。录制完成后仍须按各官方流程预处理；FTF 视觉跟踪依赖的 CUDA 阶段尚未在 Mac 验证。

## 独立传感器与检查

```bash
.venv/bin/mjpython -m wowskin_lab sensor viewer --ideal
.venv/bin/mjpython -m wowskin_lab sensor viewer --ideal --real-port auto --capture-dir outputs/calibration
.venv/bin/python -m tests unit
.venv/bin/python -m wowskin_lab official status
```

[操作与采集](docs/PICK_CUBE_ZH.md) · [ViSk / FTF](docs/LEARNING_WORKFLOWS_ZH.md) · [模型与标定](docs/MODEL_CALIBRATION_ZH.md) · [目录与测试](docs/PROJECT_LAYOUT_ZH.md) · [Pollen](docs/baselines/POLLEN_ZH.md)

**当前限制：**线材夹持可能滑动或顶开夹爪，指腹严重畸变可能触发暂停，软体计算可能慢于实时。ViSk / FTF 的完整训练与策略闭环尚未全部验证。
