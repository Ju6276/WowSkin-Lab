# Pollen 滑移检测

[Pollen 项目](https://github.com/pollen-robotics/anyskin-slip-detection) · [公开数据](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection)

输入是单块 AnySkin 的五个磁力计 XYZ，共 15 路磁场；CSV 另外提供时间和滑移标签。输出是滑移分类分数，不是位置、深度或三维力。`slip=0` 表示未标为滑移，不保证有接触或没有接触。

## 数据回放

```bash
cd WowSkin-Lab
.venv/bin/python -m wowskin_lab pollen download
.venv/bin/python -m wowskin_lab pollen evaluate
```

原数据、权重、均值/尺度和来源 manifest 位于 `assets/pollen/`。本地使用原网络结构和权重，每条记录作为长度 1 的序列；不重新拟合 scaler。结果位于 `outputs/pollen_public/`。

固定数据共 17 个 CSV、127445 帧，包含 18494 滑移帧及 108951 不滑移帧；`no_slip.csv` 为 22371 帧全零标签。已有回放结果见 [metrics.json](../../outputs/pollen_public/metrics.json)。这些数据与上游训练集重叠，不能当作独立测试或本机 WowSkin 精度。

## 仿真运行

```bash
.venv/bin/mjpython -m wowskin_lab teleop --task cable --cable-layout single_clip --input spacemouse --ideal --pollen \
  --magnetic-fit configs/magnetic_fit_pollen_nonslip.json
```

Pollen 使用左侧 15 路信号，右侧为普通硅胶。界面显示原模型分数，但不会据此自动张开夹爪；仿真与真实 AnySkin 的分布迁移未验证。

独立压头检查：

```bash
.venv/bin/mjpython -m wowskin_lab franka-bench viewer --pollen
```

P 按压、S 切向滑动、L 卸载，B 切换左/右皮肤，V 切视角，Q 退出。右侧皮肤不提供磁场。

## 与其他项目的关系

NYU Handoff Demo 是利用触觉触发交接松手的项目，Pollen 是另一个团队的滑移分类实现；不能把 NYU 某些配置中的两组输入当作 Pollen 的输入维度。[NYU 项目](https://github.com/NYU-robot-learning/AnySkin-Handoff-Demo)

Pollen 数据可以用于统计磁场范围或训练滑移分类器，没有同步力/位置/深度真值。用于当前仿真幅值匹配时，属于本地实验，见 [模型与标定](../MODEL_CALIBRATION_ZH.md)。
