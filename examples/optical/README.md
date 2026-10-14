# 光学采集示例（2026-10-14）

本机保存 12 组完整采集、共 18 条脚本示范：TACTO／FOTS-DIGIT／GelSight 各 3 条随机方块、1 条喂食、1 条刚性 single clip、1 条上下唇弹性 single clip。TACTO 六条均为不透明黑色长指部。三种后端的方块均在修正逐 episode seed 元数据后重录；喂食与卡扣保留已验证的控制器。脚本读取仿真物体姿态生成动作，不是 agent 推理或人工示范。

共 6,474 帧 LeRobot 数据；原文件约 19.0 GB，无损压缩约 7.1 GB。黑色版本的实际腕部相机截图：[方块](previews/black_cube.jpg)、[喂食](previews/black_feeding.jpg)、[弹性卡扣](previews/black_elastic_clip.jpg)。

## 本机数据包与解包

按用户要求，完整录制和约 7.1 GB 的归档不随 Git 推送；仓库只保存清单、检查结果、少量截图及工具。新 clone 不会包含这些数据，需自行采集，或从原采集机器取得对应归档。采集入口见[主 README](../../README.md) 的 `record-auto` 示例。

以下命令需要本机已有 `examples/optical/archives/` 和采集环境；`archives/` 已加入 `.gitignore`。分片是无损 tar.zst，每片不超过 512 MiB。

```bash
.venv/bin/python scripts/optical_examples.py list
.venv/bin/python scripts/optical_examples.py verify
.venv/bin/python scripts/optical_examples.py unpack --output data/examples_optical

# 也可以只解包一组
.venv/bin/python scripts/optical_examples.py unpack \
  --ids tacto_cable_insert_elastic_beam --output data/examples_optical
```

`unpack` 会先核对所有分片的大小和 SHA256，再校验解包后的每个文件；拒绝覆盖已有示例目录、路径穿越、符号链接及文件缺失。原始 H5、视频和 metadata 字节保持不变。采集时的绝对路径留作历史出处，读回优先使用随包携带的相对路径；不需要原作者的 `/Users/...` 目录。

每组目录结构为 `<id>/<task>/{raw,lerobot}`，旁边保留 `collection_report.json`，有采集控制器快照的组还保留 `controller_source.py`。LeRobot 内含三路场景 RGB、双指 RGB、Parquet 数值与无损光学 H5；本批全部包含三路场景米制深度。raw 保留 100 Hz 实测状态、已下发目标、接触与任务评分。来源配置、算法标定及 marker 缺失状态原样保存。

## 检查、回放、训练入口

```bash
.venv/bin/python scripts/optical_examples.py validate --data data/examples_optical \
  --output outputs/example_readback.json
.venv/bin/python scripts/optical_examples.py gallery --data data/examples_optical \
  --output outputs/example_gallery
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory outputs/example_gallery

# 单组也可直接用原有入口
.venv/bin/python -m wowskin_lab dataset-check \
  data/examples_optical/tacto_cable_insert_elastic_beam/cable_insert/lerobot
.venv/bin/python -m wowskin_lab dataset-review \
  data/examples_optical/tacto_cable_insert_elastic_beam/cable_insert/lerobot \
  --episode 0 --output outputs/elastic_example_review

# 方块有 3 条，可划分 2 条训练 + 1 条验证；这是接口小样本，不代表策略学会任务
.venv/bin/python -m wowskin_lab optical-act train \
  data/examples_optical/tacto_pick_cube/pick_cube/lerobot \
  --smoke --steps 100 --batch-size 4 --output outputs/black_tacto_act_example
```

所有输出目录使用尚不存在的新目录。每种喂食／卡扣配置只有一条示范，不能直接划分独立训练与验证集；需要继续采集。ACT 输入为五路 RGB＋实测机器人状态，不读取深度、marker 或物体真值。喂食从已装豆且已夹持的勺子开始；卡扣评分为中心线入槽并保持，未验收松手后的弹性锁紧。弹性材料未做实物标定，FOTS 多 body 接触仍可能没有有效 marker。这批黑色数据没有重新训练或闭环策略成功率结果。

## 检查与复现

`sources.json` 记录本机原始来源目录和任务清单，`manifest.json` 记录分片／逐文件 SHA256、帧数与源代码摘要。`verification.json` 保存换目录解包后的实际数据读回结果。完整示例归档、旧绿色录制、失败尝试、权重缓存与临时实验均保留在本机忽略目录。

本轮 88 项单元测试通过；18 条示例均已换目录解包，逐文件校验，并通过官方 LeRobot 读取、五路视频帧数、RGB-D／光学时间戳、逐行 state／下一条 action 对齐及 ACT 输入检查。弹性示例同时检查上下唇实际位移和末帧入槽保持。这些检查不代表训练后的策略成功率。

重新生成数据包的开发入口（需要本机 `sources.json` 指定的原始采集）：

```bash
.venv/bin/python scripts/optical_examples.py build --output outputs/new_optical_bundle
```
