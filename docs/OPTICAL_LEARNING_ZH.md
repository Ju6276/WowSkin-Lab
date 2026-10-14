# 光学回放与 LeRobot ACT 本地基线

ACT 来自 **Hugging Face LeRobot 0.3.3**，实现为 [`lerobot.policies.act.modeling_act.ACTPolicy`](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/policies/act/modeling_act.py)。本项目此前安装该包用于 LeRobot v2.1 数据写入／读取；现在新增本地数据、训练循环和 MuJoCo 执行适配，直接调用包内的 ACT 网络、损失、优化器配置与 checkpoint API，没有修改已安装包。它不是 TacEx／UniVTAC、ViSk 或 FTF 的训练分支，也不带已经学会本项目任务的权重。

## 输入与动作

- 每次训练对应同一任务、同一光学后端：TACTO、FOTS-DIGIT 或旧 GelSightMujoco。后端之间分目录、分 checkpoint。
- TACTO 的场景指部／支架使用不透明网格（alpha=1），长指部为黑色。来源示例 alpha=0.2 的透明外观只存在于旧数据中，不能通过更新回放页面恢复遮挡；采集时的来源 manifest 随记录保存。旧透明数据和不透明绿色长指数据均保留原样；新外观的策略应使用黑色长指部重录数据训练。
- 输入为全部场景 RGB（默认相机 1、2、52）、左／右触觉 RGB，以及 7 维实测 TCP 位姿与夹爪闭合比例。接触力、物体位置、评分、深度、marker 和语言均不作为此基线的策略输入。
- 训练读取 LeRobot 编码视频，在线读取新渲染 RGB；均按长边缩放、灰色 127 补边，使用 OpenCV `INTER_AREA`、除以 255 与 ImageNet 归一化。没有调整触觉亮度；有损视频与在线图像仍存在压缩差异。
- 原始 state 是实测反馈；action 是观测后第一条实际 100 Hz 控制指令的**绝对**基座系 TCP 目标和闭合比例，不是实测下一状态或位移增量。
- TCP 常处于基座系约 180° 姿态。适配层将 state 和 action 的旋转都表达为相对第一个训练观测姿态的旋转向量，避免 `+π/-π` 跳变；参考姿态保存在 `run.json`。执行前还原绝对旋转，原始数据不变。演示接近新的相对旋转分支边界时明确拒绝；这套 7 维表示不是任意大旋转的通用解法。
- 策略 30 Hz、控制 100 Hz，按 4／3／3 个真实 tick 推进。ACT 预测动作块，每次执行第一个目标后重新观察。绝对目标减去尚在执行的目标，再转为世界平移／TCP 局部旋转增量；不重复累加，不用实测位置重置目标来绕过伺服限制。
- 仅用 `phase=1` 的演示区间；默认最后一个完整 episode 验证，其余训练。统计量仅用训练 episode。末尾动作块在同一 episode 内填充，通过官方 `action_is_pad` 排除填充损失。至少需要两个 episode。

## 离线回放

```bash
.venv/bin/python -m wowskin_lab dataset-review \
  data/episodes_tacto/pick_cube/lerobot --episode 0 \
  --output outputs/review_tacto_000
```

也可传入单个 raw `capture_XXXXXX` 目录检查失败／未审核段。保留 raw 数值附件和引用媒体；只有 LeRobot 而丢失 raw 时，不能恢复真实 100 Hz 曲线，命令会报错。

输出 `index.html`、`review.mp4`、`review.json`，放在同一目录。页面同步显示三路场景 RGB、双指 RGB／压入深度／接触掩码／marker、法向力、人工与自动评分、100 Hz 实测与已下发目标曲线，支持播放、调速、逐帧和拖动进度。图像对应实际采样时间，不用 `frame/30` 代替真实时刻。压入显示为 mm，双指全段同一色标；H5 仍是米。缺少 marker 时显示实际 `marker_status`，不会补零伪造测量。

视频内嵌在 HTML 中，并将解码帧绘制到页面画布；播放不依赖网络，也不依赖本地服务支持 HTTP Range。MP4／JSON 仍单独保留以便下载、检查。若浏览器限制打开本地 HTML，可仅在本机启动：

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory outputs/review_tacto_000
# 浏览器打开 http://127.0.0.1:8765/
```

MP4 是缩放、编码的检查副本，精确数组仍在源光学 H5。页面不显示场景相机的全部深度附件。

本机回放入口为 `outputs/optical_replays_20261014/index.html`，可切换任务、TACTO／FOTS-DIGIT／GelSight 及示范。保留九条抓方块示范，新增三种后端各一条喂食和 single clip，另新增三条上下唇弹性 single clip，共十八条；卡扣模型可独立选择，TACTO 使用不透明外壳。目录含独立 HTML／MP4／JSON 副本，可用 `python -m http.server 8765 --bind 127.0.0.1 --directory outputs/optical_replays_20261014` 启动总入口。完整数据包和生成的回放均保留在本机忽略目录，不随仓库推送；[examples/optical](../examples/optical/README.md) 保存清单、检查结果和截图，可在持有本机数据包时解包并重建回放。

## 小样本流程

输出目录必须尚不存在。方块脚本读取初始方块位置；喂食与 single clip 脚本持续读取仿真物体姿态生成参考动作，训练和策略执行不接收这些真值。`--no-record-depth` 仅省略场景深度，双指光学深度仍保存；自动和手动采集默认均保存场景深度。

```bash
.venv/bin/python -m wowskin_lab record-auto --sensor-backend tacto \
  --output outputs/tacto_trial --episodes 3 --randomize --no-record-depth
.venv/bin/python -m wowskin_lab dataset-check outputs/tacto_trial/pick_cube/lerobot
.venv/bin/python -m wowskin_lab optical-act train \
  outputs/tacto_trial/pick_cube/lerobot --output outputs/tacto_act_trial \
  --smoke --steps 500 --batch-size 8 --lr 0.0001 --device mps
.venv/bin/python -m wowskin_lab optical-act eval outputs/tacto_act_trial \
  --output outputs/tacto_eval_trial --device mps --seed 100 --episodes 1 --max-steps 450
```

切换 `--sensor-backend digit_fots|gelsight` 并给每次采集、训练、评估使用新目录；设备可选 `cpu|mps|cuda`。

喂食和 single clip 的采集命令见 [自动采集](COLLECTION_ZH.md#自动采集)。本次每种配置只有一条演示，可用于 `dataset-check`、`dataset-review` 和训练数据读取检查；ACT 的训练／验证分割至少需要两条。不能把同一条演示同时作为训练和验证数据。

`--smoke` 使用 96×96 图像、随机初始化 ResNet18、128 维 transformer、10 步动作块验证接口，不是论文默认配置。去掉该参数使用 224×224、官方 ACT 默认网络宽度与 ImageNet ResNet18 初始化、30 步动作块，首次可能下载主干权重。步数与学习率需要根据数据再确定；命令不保证任务学会。

训练保留 `run.json`（来源、分割、相对姿态参考、配置、版本、指标）、`loss.jsonl` 和官方格式 `pretrained_model/`。`image_cache/` 是训练尺寸图像磁盘缓存，内存映射读取。当前命令每次新建训练，不提供断点续训。数据有 `INCOMPLETE` 标记时拒绝训练。

评估从 checkpoint 场景配置重建环境，读取实时渲染与实测状态，不读训练视频、状态序列或示范动作；输出逐步观测／预测／实际命令、预览视频和任务评分。载入时检查 checkpoint 哈希、LeRobot／MuJoCo 版本及训练时环境／光学／相机配置源码摘要。这不能追溯旧数据采集时尚未保存的全部代码、资产版本。

## 实测范围

2026-10-14 的实验位于 `outputs/optical_pipeline_20261013/`（实验在前一天开始）。三种后端各录制三个随机初始位置的完整抓放，共九条，全部通过自动抓取、搬运、放下、松爪评分。这是脚本示范，不是九次学习策略成功。

各后端各用 2 条训练、1 条验证，compact ACT 在 MPS 上各训练 500 步，batch=8、学习率 `1e-4`。验证集 XYZ 绝对误差下降到各轴约 10–18 mm，但三种后端各一次、seed=100、最多 450 个策略步的真实闭环均为 **0/1 成功**，均未满足抓起评分。这轮证明接口可运行，未证明策略已学会任务，不能据此宣布批量采集与训练已全部验收。

各后端 `*_act_v2/run.json` 与 `*_eval_v2/report.json` 保存准确结果；`status=complete` 仅表示流程正常结束，任务结果看 `successes`／episode 的 `success`。实际检查：83 项单元测试、8 项数据／任务接入回归、1 项覆盖三后端的实际录制／导出／回放／训练读取回归均通过，无跳过；Safari 已实际验证播放、暂停、逐帧、跳转及真实时钟。没有重新运行仓库完整测试套件。

回归覆盖跨 `±π` 姿态、绝对目标不重复累加、30／100 Hz 调度、RGB 补边、末尾掩码、训练／验证隔离、三个后端实际录制和回放。喂食和 single clip 后续已各补一条示范／后端（见下文），尚未训练或闭环验收；绕线和插孔仍需完整示范及后续验证。薯片没有自动任务评分，此 `eval` 入口不接入。尚无触觉消融、跨 Isaac／MuJoCo 或真机迁移结论。

2026-10-14 后续外观修正：TACTO 各任务的指部／支架改为不透明。空载与夹持的同状态对照确认三路场景 RGB 发生遮挡变化，而左右触觉 RGB、压入深度、接触掩码和法向力逐项相同。源机械参数／安装方向的两项检查、覆盖 18 个任务组合的接入检查及覆盖三后端的录制回归通过。新目录 `outputs/optical_opaque_20261014/tacto/` 重录三条随机方块示范，3/3 通过自动抓放评分与数据检查；`outputs/optical_opaque_20261014/tacto_review/` 为第一条的 370 帧回放。新记录的来源 manifest 写明了外观适配。旧 `optical_pipeline_20261013` 数据保留，以上 ACT 结果仍属于旧透明数据；本次未重新训练 ACT。

2026-10-14 喂食／single clip 后续采集：`outputs/optical_tasks_20261014/` 保存三种后端各一条喂食和单卡扣示范，正式目录分别为 `{backend}_feeding` 和 `{backend}_single_clip_v2`。三路场景深度开启；采集使用规则控制器和物体真值，不是学习策略。喂食从场景初始化的已夹持装豆勺子开始，三条均送到嘴前区域，收尾保留 6 颗豆子且没有头部接触；不包含取勺、装豆或吞咽。single clip 使用默认刚性卡扣，三条收尾时均跨过槽内三个截面，连续入槽约 3.5 秒；没有松爪，不代表卡扣锁紧或松手后保持。

首轮 FOTS／GelSight 单卡扣数据虽触发了保留式成功评分，但在停止移动后滑出槽。控制器现已改为成功后继续居中并等待稳定，再录制收尾；回放和正式验收采用三种后端的 `_v2` 记录，首轮目录仅保留作诊断。不能仅凭 `success=true` 推定末帧仍满足任务条件。

这六条逐一通过官方 LeRobot 读取、全部视频帧数、全部数值有限性、逐帧场景深度／触觉时间戳、每行 state 与下一条实际 action 对齐检查；ACT 读取和 episode 末尾动作掩码也通过。喂食帧数为 299／297／297，single clip 为 356／368／372（均按 TACTO／FOTS／GelSight 排序，含预录和后录）。20 项相关单元回归通过、无跳过；没有重跑完整仓库测试，也没有训练这两项任务。检查汇总及六个准确数据路径保存在 `outputs/optical_tasks_20261014/verification.json`，逐条结果保存在各正式目录的 `verification.json`。

字段可用性边界：FOTS 的 single clip 在左指 152／368 帧、右指 180／368 帧标记为 `multiple_bodies_unsupported`，这些帧没有有效 marker 流；线材多节 body 接触触发了来源算法的限制。喂食 FOTS 未出现这个状态。TACTO／旧 GelSight 的 marker 均为 `not_provided`。这些状态原样保留，没有补造 marker；当前 ACT 只读双指 RGB，不使用 marker。各指完整状态计数见上述检查汇总。


2026-10-14 上下唇弹性采集：`record-auto --task cable --cable-layout single_clip --clip-model elastic_beam` 已接入已有 `experimental_discrete_elastic_clip_v2_two_lips` 模型。安装底座固定，上下唇被动弯曲；使用原有弹性卡扣配置和光学任务的 discrete 积分适配，没有改变材料参数、碰撞参数或评分来获得成功。记录在 `outputs/optical_elastic_20261014/{tacto,digit_fots,gelsight}/cable_insert/`，与原刚性记录分开。为适配不同指部几何，脚本采用后端对应的抓取位置、推进速度和稳定收尾方式；准确参数和控制器源码快照随采集保存。

三条分别为 394／378／383 帧，均通过现有入槽评分，收尾仍在槽内并已连续保持约 3.5 秒，数值警告为零。上下唇尖初始距离均约 2.626 mm，最大距离分别约 9.997／7.974／8.402 mm；上唇尖最大位移为 9.658／8.079／8.652 mm，下唇尖为 0.346／0.442／0.294 mm。位移相对于各条记录首帧，以上均按 TACTO／FOTS／GelSight 排序。回放新增弹性卡扣标签、开口和上下唇尖位移曲线，数值逐项对照原始 100 Hz H5，而非从视频估计。

三条已检查官方读取、五路视频全部帧数、RGB／深度／触觉时钟、每行 state 与实际下一条 action 对齐及 ACT 数据读取；17 项相关测试通过、无跳过，未运行完整仓库套件。结果在 `outputs/optical_elastic_20261014/verification.json`。没有测试松爪后的保持，也没有训练这批数据。FOTS 弹性插扣的多 body 接触仍不支持 marker（左右分别 224／378 和 193／378 帧），缺失状态原样保存；其余后端不提供 marker。

2026-10-14 长指部颜色修正：TACTO 场景可视网格 `finger_0` 由绿色改为黑色，alpha 仍为 1；内侧网格保留来源颜色。这次单独改色时尚未重录十八条回放，其中 TACTO 长指仍为绿色；后续重录见下文。本次修改不回填旧录像或采集 metadata。关闭随机光学噪声后，方块空载／实际夹持及喂食已夹勺状态的同状态对照中，三路场景相机均出现改色，双指触觉 RGB、压入深度、接触掩码和法向力逐项相同；TACTO 来源文件 SHA256 仍匹配。对照截图与报告在本机 `outputs/black_fingers_20261014/`。

2026-10-14 提交前完整回归：`.venv/bin/python -m tests all` 实际运行 227 项，耗时 845.191 s；220 项通过、1 项错误、6 项跳过，进程退出码为 1。分组为 unit 85 项通过、physics 109 项通过／1 项错误、official 18 项通过、artifacts 8 项通过／6 项跳过。唯一错误为 `tests.physics.test_robotiq.RobotiqChecks.test_source_servo_and_contact_release`：Robotiq＋AnySkin 强夹线在 3.420 s 出现负体积比（左右约 −0.643／−0.573），仍是已记录的物理问题；不是光学颜色或录制接口通过就已解决的问题。

本次六项跳过均为可选资源条件，没有 MuJoCo 旧版本跳过：

- `ReskinForwardChecks.test_saved_predictions_and_spatial_split`、`test_unsupported_input_is_not_silently_extrapolated`：缺少可选 3D Cal 正向模型。
- `InverseChecks.test_forward_original_and_mapped_archives_are_preserved`：缺少原始／映射正向模型存档元数据。
- `PseudoLabelArtifactChecks.test_baseline_and_public_data_preserved_and_no_heldout_training`、`test_saved_models_reproduce_report_without_claiming_measured_pseudo_depths`、`test_source_only_control_saved_metrics`：未运行可选伪标签实验。

完整日志保存在本机 `outputs/black_fingers_20261014/tests_all.log`。另已检查采集／回放／训练命令的 `--help`、Python 编译及 `git diff --check`。这些结果包含真实光学录制、选集导出、五路视频读取、回放与 ACT 数据适配检查，不代表重新训练或策略任务成功率验收。

2026-10-14 黑色重采与本机示例整理：TACTO 三条随机方块、喂食、刚性单卡扣、弹性单卡扣共六条均已用黑色长指部重新采集。本轮发现方块脚本的逐 episode seed 未同步进 raw metadata，已修正 `auto_collect.py`，并重录三个后端各三条方块，实际 reset seed 和 metadata 均为 0／1／2；方块采集还新增控制器代码快照与 SHA256。旧录制和失败尝试保留在本机，不放入本批示例。当前十八条完整示例按十二组保留在本机，全部包含 RGB-D、双指无损光学数据、100 Hz raw 数值及 LeRobot。本机分片、逐文件校验、解包和重建回放入口见[示例文档](../examples/optical/README.md)。本轮没有重新训练策略。

本轮实际运行 88 项单元测试，全部通过；十八条本机数据在换目录解包后逐文件 SHA256 一致，全部通过官方读取、五路视频帧数、时间戳、逐行状态／下一条动作和 ACT 输入检查。回放已切换为当前黑色版本。结果保存在 `examples/optical/verification.json`；未重复运行上文的完整物理套件，Robotiq＋AnySkin 的已知问题仍保留。

2026-10-14 按用户要求取消大数据上传：完整归档从待推送提交中移除，保留在本机；仅提交清单、检查结果、预览图和复现工具。上传及自动 push 进程已停止。取消前已有部分 LFS 对象传到远端，停止上传和移除 Git 指针不会自动删除这些对象。
