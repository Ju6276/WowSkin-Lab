# 论文、传感器与可复用算法

这里记录方法的用途与本项目接入边界。安装与操作见 [README](../README.md)，固定源码版本和许可见 [THIRD_PARTY](../THIRD_PARTY.md)，本地参数见 [模型说明](MODEL_CALIBRATION_ZH.md)。论文结果不作为本项目验证结果。

## 已接入的光学传感器

### TACTO / DIGIT

[论文](https://arxiv.org/abs/2012.08456) · [原仓库](https://github.com/facebookresearch/tacto) · [MuJoCo 移植](https://github.com/utn-air/mujoco-tacto)

物理引擎提供接触和物体位姿，TACTO 生成光学触觉 RGB 与深度。本项目使用固定 MuJoCo 移植的 DIGIT 机械模型及原 Renderer，接入独立按压台、Panda 任务与双指录制；它不求解真实胶体 FEM，也不提供 FOTS 的 marker motion。

[抓取稳定性实验](https://github.com/facebookresearch/tacto/tree/main/experiments/grasp_stability)可参考视觉／触觉融合分类。该任务估计抓稳概率，不能直接作为本项目的 7 维动作策略；数据生成环境、标签与输入需要单独适配。目前未接入该分类器。

### FOTS / DIGIT

[论文](https://arxiv.org/abs/2404.19217) · [仓库](https://github.com/Rancho-zhao/FOTS) · [本项目固定 MuJoCo 分支](https://github.com/Rancho-zhao/FOTS/tree/4bd82cfc75ad8d600b2008df513b28cf04611610)

FOTS 是光学仿真方法，不是传感器硬件名称。当前接入作者 MuJoCo 分支的 DIGIT：原机械模块、内部相机、MLP RGB、阴影以及标记点膨胀／剪切／扭转。它与 TACTO 的机械模型和深度定义分别保留，不能视作同一物理状态下仅切换渲染函数。

所接入示例的 marker 剪切／扭转使用单条刚体运动轨迹；多个刚体同时接触时，当前显示 marker 不支持，仍输出 RGB／深度。没有本地多物体流叠加。公开渲染代码可用于观测生成；本项目未复现论文的机器人策略实验。

### GelSightMujoco

[源仓库](https://github.com/rlamprell/GelSightMujoco) · [源光学驱动](https://github.com/rlamprell/GelSightMujoco/blob/main/Test_bed/envs/utils/gelsight_driver.py)

原 mountable GelSight 模型使用内部深度、背景、法向光照与阴影生成 RGB。本项目加载完整源模块并适配 Panda，保留源光学函数；这是旧款模型，不是 GelSight Mini，也不是 FOTS 的 GelSight 标定。

源项目还包含旧 Gym／TensorFlow 的 DDQN、HDDQN 操作代码；机器人、动作和依赖不同，未移植到本项目。当前大平面铺满内部视野时 RGB 变化可能很小：源函数一致性不等于触觉信息足够，需检查深度、局部接触和下游可辨识性。

## 可选的光学学习路线：Sparsh

[论文](https://arxiv.org/abs/2410.24090) · [官方代码](https://github.com/facebookresearch/sparsh) · [项目](https://sparsh-ssl.github.io/) · [预训练权重](https://huggingface.co/collections/facebook/sparsh)

Sparsh 用 MAE、DINO、JEPA 等自监督方法训练触觉图像编码器，覆盖 DIGIT、GelSight '17、GelSight Mini。输入触觉图像或按配置组织的序列，输出特征；下游再接力、滑移、位姿或动作预测模块。编码器本身不是仿真器，也不会直接控制 Franka。

官方真实触觉数据用于预训练和评估。使用现成权重不要求先自行采集完整预训练数据；在本项目中训练操作策略仍需任务示范和动作头。可尝试双指光学 RGB → Sparsh 特征 → 与场景 RGB／机器人状态融合 → 动作，但这是待实现方案。真实预训练图像与仿真图像有分布差异；编码器无法补回观测中没有的信息。当前没有接入 Sparsh，也不将其算作传感器测试通过的一部分。

## 磁场传感器与已接入学习流程

- **AnySkin**：[论文](https://arxiv.org/abs/2409.08276)。磁化软皮肤变形影响五个三轴磁力计。本项目的弹性网格＋磁偶极子是本地近似，不是作者官方仿真器。
- **ViSk / VisuoSkin**：[项目](https://visuoskin.github.io/) · [代码](https://github.com/raunaqbhirangi/visuoskin)。已接入固定版转换、训练和闭环入口；本地三相机、单块 15 路磁场、6 Hz 时序和仿真控制有明确适配。不能把当前光学图像直接送入磁场版。
- **Feel the Force**：[代码](https://github.com/feel-the-force-ftf/feel-the-force)。已接数据导出和标注，视觉跟踪与完整策略闭环尚未完成。具体步骤和要求见 [学习工作流](LEARNING_WORKFLOWS_ZH.md)。
- **3D Cal / ReSkin**：[论文](https://arxiv.org/abs/2511.03078) · [数据](https://zenodo.org/records/18462608) · [代码](https://github.com/rohankotanu/py3DCal)。提供位置／压深与磁场配对数据；官方 MagNet 是磁场到位置／深度的逆模型，本项目的小前向 MLP 是本地训练，WowSkin 迁移尚未验证。

## 候选物理求解与任务参考

- **TacIPC**：[论文](https://arxiv.org/abs/2311.05843) · [代码](https://github.com/Ailon-Island/TacIPC)。FEM＋IPC 弹性接触，可借鉴固定边界、接触与防翻转；未接入。原项目依赖 CUDA，不能直接在 Apple Silicon 上运行。
- **DiffTactile**：[论文](https://arxiv.org/abs/2403.08716) · [代码](https://github.com/Genesis-Embodied-AI/DiffTactile)。可微接触、材料模型与参数辨识参考；未接入，磁场观测仍需独立实现。
- **UniVTAC**：[代码](https://github.com/univtac/UniVTAC)。参考视触觉任务、示范、训练与评估组织。
- **NeoSim**：[代码](https://github.com/neoteai/N0-Foundation/tree/main/neosim)。参考精细插入与脆弱物体操作；`grasp_chip` 的损伤判据与本地可断连接模型不同，见模型说明。
- **SoftVTBench**：[代码](https://github.com/TuojingAI/SoftVTBench)。参考软物体任务中任务完成与安全完成的区分。
- **ContactWorld**：[代码](https://github.com/PokuangZhou/ContactWorld)。参考视触觉世界模型和规划的输入／评估方式。

最后四项作为任务与学习参考保留，尚未接入本项目，不称为当前 MuJoCo 后端的 benchmark。具体未完成工作集中在 [TODO](../TODO.md)。
