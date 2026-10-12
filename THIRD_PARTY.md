# 第三方来源与再分发

本项目的本地接口和任务配置与上游实现不同。引用第三方算法不表示获得论文的实验结果。当前没有为原创代码选择根目录 LICENSE；准备公开发布前，由仓库所有者确定授权，并逐项核对素材条款。这里记录来源，不替上游重新授权。

## 固定源码依赖

[configs/upstream.json](configs/upstream.json) 保存 URL 与提交号。`scripts/setup_upstream.py` 下载到被 Git 忽略的 `third_party/`：

- `franka_xbox`：Franka 模型／素材；是日常场景的安装依赖。
- `visuoskin`：ViSk 原转换器、加载器、训练与推理；仅在独立工作目录应用记录在 manifest 中的数据／环境适配，参考 checkout 不修改。
- `Open-Teach`：ViSk 原版 XArm 环境导入的网络工具；随 `--only visk` 一起获取。
- `feel-the-force`：FTF 及其固定子模块，遵循各自许可证。
- `mujoco-tacto`：可选 DIGIT 光学渲染器与独立传感器 STL（MIT）。参考 checkout 不修改；STL 连通分量在加载时分离外壳和曲面胶体，附加到原手指末端。接触／坐标、实际曲面光学几何与 Mac CGL 接入在本地模块，初始化代码的许可见 [TACTO 记录](assets/reference/tacto/manifest.json) 和 [LICENSE](assets/reference/tacto/LICENSE)。pyrender 使用 RobotControlStack 的固定提交，见 `requirements/tacto.txt`。
- `FOTS`：主分支 GelSight 光学 MLP、背景、像素标定、平面阴影与标记点算法，[固定版本及文件摘要](assets/reference/fots/manifest.json)。运行时加载原源码；CPU 权重加载、私有导入、NumPy 2 兼容、接触历史及多物体位移叠加属于本地接口。作者 README 声明 MIT，但该提交没有实际 `LICENSE` 文件；不将源码、标定图和权重重新打包分发。作者 `FOTS-mujoco` 分支使用 DIGIT，不能作为本地 GelSight 外壳的来源。

参考 checkout 不随本仓库提交；各自的 LICENSE / NOTICE 保留在下载目录中。

## 已包含的素材与代码

- pyrender 正交深度读取：[来源与改动](assets/reference/pyrender/manifest.json)、[MIT 许可证](assets/reference/pyrender/LICENSE)。固定上游版对所有相机使用透视深度反投影；本地 FOTS 深度渲染器使用正交投影对应的线性深度读取，不改变 TACTO 渲染器。
- SERL++ 可选笛卡尔阻抗：[源码与摘要](assets/reference/serl_controller/manifest.json)、[Apache-2.0 许可证](assets/reference/serl_controller/LICENSE)。保留原文件；运行时只适配 MuJoCo `mj_fullM` 调用接口，初始化和输入处理属于本地接入。
- SERL++ Peg Insert：[来源与逐文件 SHA256](assets/serl_peg/manifest.json)、[上游许可证](assets/serl_peg/LICENSE)。网格、凸分解与源 XML 保持原样；本地场景不复制 grasp weld，使用自己的机器人、抓法、控制与评分。
- EBiM 线材、卡扣与 Robotiq：[来源 manifest](assets/ebim/manifest.json)、[许可证](assets/ebim/LICENSE)、[NOTICE](assets/ebim/NOTICE)。
- EBiM 喂食：[来源 manifest](assets/ebim_feeding/manifest.json)、[许可证](assets/ebim_feeding/LICENSE)、[NOTICE](assets/ebim_feeding/NOTICE)、[上游说明](assets/ebim_feeding/upstream_README.md)。派生人体／布局不等于上游完整任务。
- Panda 执行器参考：[Menagerie manifest](assets/reference/panda_menagerie/manifest.json) 与 [LICENSE](assets/reference/panda_menagerie/LICENSE)；[robosuite manifest](assets/reference/robosuite_panda/manifest.json) 与 [LICENSE](assets/reference/robosuite_panda/LICENSE)。
- WowSkin 原厂可视化：[manifest](third_party/wowskin_visualizer/manifest.json)、[LICENSE](third_party/wowskin_visualizer/LICENSE)。
- 鞭子参考模型：[manifest](assets/whip_reference/manifest.json)、[LICENSE](assets/whip_reference/LICENSE)。不是当前 cable 任务的线材后端。
- 3D Cal / MagNet 源文件：[manifest](assets/3dcal/upstream/manifest.json)、[LICENSE](assets/3dcal/upstream/LICENSE)。官方 ReSkin 数据与预训练权重按需下载，不随当前源码分发；[数据来源](assets/3dcal/manifest.json)与权重摘要保留在 manifest 中，不能由源码许可证推定数据许可。获取命令见[学习工作流](docs/LEARNING_WORKFLOWS_ZH.md#可选触觉参考模型)。
- AnySkin CAD：[文件来源与 SHA256](assets/cad/manifest.json)、[几何说明](assets/cad/README_ZH.md)。来源清单未包含明确再分发许可，公开前须核对。

`assets/` 中的原始文件、派生文件及用途应按所在目录 manifest 区分。保留作者、许可与来源，不因文件移动或格式转换删除归属信息。

## 不随代码发布的内容

`data/`、`outputs/`、虚拟环境、可选 Pollen 下载、本地训练 checkpoint、标定记录、W&B 凭据不进入 Git。示例命令中的数据／权重路径由使用者生成，不能假设 clone 后已有。

Pollen 数据与模型来源见 [Pollen 说明](docs/baselines/POLLEN_ZH.md)。本地 ReSkin 前向网络、逆模型与 Pollen 幅值拟合是研究对照，默认拟合偶极子只需要已提交的增益配置，不要求这些可选权重存在。
