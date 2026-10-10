# WowSkin-Lab

Franka 的 MuJoCo 遥操作与触觉采集：左指 WowSkin、右指普通硅胶，支持键盘、SpaceMouse 和 Pico。采集固定使用两台外部相机，从不同角度观察同一桌面区域。采集与训练流程面向 ViSk、FTF；真实 Franka 后端尚未接入。

## 安装（Mac）

```bash
git clone git@github.com:Ju6276/WowSkin-Lab.git
cd WowSkin-Lab
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements-acquisition.txt
.venv/bin/python scripts/setup_upstream.py --only franka
```

已安装的本机环境可直接启动。官方算法源码另用 `scripts/setup_upstream.py --only visk ftf` 下载，训练依赖和数据处理见[学习流程](docs/LEARNING_WORKFLOWS_ZH.md)。用户采集数据、本地训练权重、虚拟环境和下载的上游仓库不纳入 Git；仓库保留有来源说明的公开标定样本与参考文件。

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

键盘测试将 `--input spacemouse` 改为 `--input keyboard`；Pico 改为 `--input pico`，连接设置见[操作说明](docs/PICK_CUBE_ZH.md)。喂食从夹住装有豆子的勺子开始；复位用 Newton 建立接触，遥操作优先使用稀疏 CG；若失稳，回滚该次计算并改用 Newton，窗口显示当前求解器。物理计算与窗口显示分开运行；复位时显示接触稳定过程，按 **P** 可中断退出。窗口刷新不代表物理已达到实时速度。

三个任务默认使用 **MuJoCo 形变＋偶极子＋Pollen/网络拟合增益**，加载 `configs/magnetic_fit_pollen_nonslip.json`。它是仿真幅值拟合，不是实物标定；方块和勺子的迁移尚未验证。直接网络预览用 `--reskin-model outputs/reskin_forward_mapped --reskin-extrapolate`，存在超出训练范围的问题。窗口明确标注当前方法。

## 操作与录制

右侧显示五点触觉和两台采集相机的实时预览；预览不启动录制。

先点击窗口。SpaceMouse 推拉/倾斜控制六轴，左键夹紧、右键张开、双键保持开度。

- 键盘：W/S 前后，A/D 左右，E/Q 上下；I/K、L/J、U/O 旋转；X/Z 夹紧/张开。
- **B** 复位并开始录制，先保持无接触约 1 秒仿真时间。
- **Enter / N** 保存成功 / 失败；**R** 复位；**V** 切视角；**P** 保存中断记录并退出。

当前默认拟合模式的双格式记录保存在 `outputs/diagnostics/fitted_dipole_preview/two_camera/episodes/`。历史原始偶极子记录仍在 `data/official_two_camera/`；不会覆盖。录制完成后仍须按各官方流程预处理；FTF 视觉跟踪依赖的 CUDA 阶段尚未在 Mac 验证。

## 独立传感器与检查

```bash
.venv/bin/mjpython -m wowskin_lab sensor viewer --ideal
.venv/bin/mjpython -m wowskin_lab sensor viewer --ideal --real-port auto --capture-dir outputs/calibration
.venv/bin/python -m tests unit
.venv/bin/python -m wowskin_lab official status
```

[操作与采集](docs/PICK_CUBE_ZH.md) · [ViSk / FTF](docs/LEARNING_WORKFLOWS_ZH.md) · [模型与标定](docs/MODEL_CALIBRATION_ZH.md) · [目录与测试](docs/PROJECT_LAYOUT_ZH.md) · [Pollen](docs/baselines/POLLEN_ZH.md)

**当前限制：**线材夹持可能滑动或顶开夹爪，指腹严重畸变可能触发暂停，软体计算可能慢于实时。ViSk / FTF 的完整训练与策略闭环尚未全部验证。
