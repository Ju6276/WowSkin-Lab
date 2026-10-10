# 遥操作与数据采集

所有启动命令见 [README](../README.md)。在 项目根目录 运行，每次只启动一个遥操作窗口。图形入口使用 `.venv/bin/mjpython`。

## 场景选择

- `--task pick_cube`：夹取方块；追加 `--randomize` 可在复位时将方块 XY 各随机偏移 ±25 mm，`--seed` 控制序列。成功条件是双侧接触且抬高超过 60 mm，没有指定位置的放置判据。
- `--task cable`：原绕线布局，两个导向柱、右侧黄色卡槽及固定座。
- `--task cable --cable-layout single_clip`：简单插线，保留线材固定座和一个黄色 C 型卡槽；卡槽旋转 180°。卡槽位于初始线束的 +Y 一侧，body 位置 `(0.45,-0.06,0.1009) m`，入口距初始线束约 5 cm；线材首段仍固定。数据任务名 `cable_insert`。
- `--task feeding`：固定机器人送食，勺子已夹住并装有六颗豆子；人头在桌面，其他餐具已移除。显示头部接触载荷及峰值；没有口腔/吞咽模型或自动成功判定。

线材和喂食使用本地布局及 [EBiM 原素材](../assets/ebim/manifest.json) / [喂食素材](../assets/ebim_feeding/manifest.json)，不代表完整官方评测任务。卡槽网格、碰撞块和线材材料参数保留来源配置；短线长度为 0.38 m。简单布局不改变原绕线场景。

## 控制与视角

先点击窗口。W/S 或上下箭头沿画面前后移动；A/D 或左右箭头沿画面左右移动；E/Q 上下。I/K、L/J、U/O 分别绕末端局部 X/Y/Z 轴旋转。支持长按，松开停止。

X/Z 闭合/张开，C 保持当前开度。方块/喂食默认采用 Menagerie Panda 位置伺服；执行器的 ±100 N 是上限而非持续夹持力。线材采用 [robosuite Panda](../assets/reference/robosuite_panda/manifest.json) 的每指独立位置伺服：1000 N/m、每指 ±20 N，并保留其关节阻尼、惯量和摩擦参数；本地安装位置及行程另行保留。这是基准控制参数，不是真机接触力标定。`--grip-force` 才显式启用旧力限制器；默认的 -/= 不调力。

V 切换近景/全景，Esc 恢复全景；鼠标左拖旋转、右拖平移、滚轮缩放。逗号/句号调整触觉显示比例。R/Backspace 复位，P 保存并退出。F 仅用于方块自动短测，不能用于线材或喂食。

若显示 `PAUSED: physics invalid`，物理求解已停止，触觉面板保留最后有效帧；按 R 复位后继续。该段录制会中止，不导出失效后的物理数据。

## SpaceMouse USB 遥操作

使用 `--input spacemouse`。旋钮推拉映射为相机平面 XY，提起为世界 Z，倾斜/扭转控制三维旋转。左键夹紧、右键张开、双键保持开度。旋钮回中停止，失去窗口焦点停止输入。

默认平移上限为 0.12 m/s、角速度 0.5 rad/s，均按仿真时间计。可设置 `--spacemouse-speed`、`--spacemouse-angular-speed`、`--spacemouse-deadzone`。面板的 `Simulation speed` 是实际实时倍率；换输入设备不会减少软体求解耗时。

## Pico USB 遥操作

使用 `--input pico`，USB 连接头显并开启 USB 调试。程序建立 ADB 转发并打开头显浏览器；选择佩戴/对面模式，点击“进入遥操作”，先松开一次右握键。

- 右握键：按住相对移动/旋转，松开保持，再握住建立新起点。
- 右扳机：握键按住时调夹爪闭合度。
- 右 B：握键松开时切换佩戴/对面模式。
- 右 A：握键松开、手柄水平指向屏幕时重设操作前方。
- 左 X：开始录制或结束为待审核；左 Y 保存成功；按下左摇杆保存失败。录制按钮操作前松开右握键。

对面模式只在 Mac 显示完整仿真，头显仍需保持唤醒和 WebXR 追踪。追踪中断或重定位后，松开再握住右握键恢复。右手控制单条 Franka，左手用于录制；此记录属于 FTF 机器人分支。

连接排查：`.tools/platform-tools/adb devices -l` 应显示一台已授权设备。服务重启后重新进入 VR；旧页面连接失败时刷新页面。

## 五个点的 XYZ 怎么看

面板显示五个磁力计的三轴差分磁场，单位 μT；绿色线段表示 XY，红色圆圈表示 Z 的符号与幅值（正值实心、负值空心）。默认比例为 7 μT/逻辑像素。它不是三维力。

原偶极子与拟合偶极子使用 U1中心、U2上、U3右、U4下、U5左的 PCB 参考顺序。网络预览使用 CH1中心、CH2左、CH3右、CH4上、CH5下的候选分组。实物对应与轴向仍未全部验证。

`--magnetic-fit configs/magnetic_fit_pollen_nonslip.json` 启用拟合偶极子；`--reskin-model outputs/reskin_forward_mapped --reskin-extrapolate` 启用网络预览，两个选项互斥。详见 [模型与标定](MODEL_CALIBRATION_ZH.md)。

## 保存了哪些数据

使用双格式录制：线材/喂食默认开启，方块需 `--record-format both-raw`。

1. B 复位并开始一段示教，保持无接触约 1 秒仿真时间。
2. 遥操作完成任务。
3. Enter 保存成功，N 保存失败；再次 B 保存待审核；P 保存中断并退出。

Enter 是操作者标记，线材/喂食没有自动成功评分。未经审核或失败的段不会自动作为成功专家示教使用。

- ViSk：两路固定相机图像、实际位姿与夹爪、七维命令、15 路原始磁场、基线、时间戳和配置。
- FTF：官方 Franka-Teach 写出的双路视频、状态/动作 PKL、触觉 H5 与元数据。
- 两套数据通过 `paired_recording_id` 关联；状态/触觉为 100 Hz 仿真时钟，相机约 30 Hz。

默认根目录为 `data/official_two_camera/`，其下 `visk/raw/<任务>/` 与 `ftf-robot/extracted_data/<任务>/` 分开保存。`--record-dir` 可指定另一根目录。启用拟合或网络预览时，默认改用 `outputs/diagnostics/fitted_dipole_preview/two_camera/episodes/` 或 `reskin_preview/two_camera/episodes/`，其下仍按任务分开。

全会话数值诊断在 `outputs/<任务>/teleop_*/`，不能替代包含图像的正式示教。FTF 还需要官方视觉点标注与跟踪，录制后处理见 [学习工作流](LEARNING_WORKFLOWS_ZH.md)。

## 自动采集

```bash
.venv/bin/python -m wowskin_lab record-auto --output data/automated/my_run --episodes 2 --randomize
```

使用新目录。该命令是方块脚本示教，不能用作线材/喂食自动成功验证。

## 使用范围

方块保持段的滑移回归未通过，线材夹爪仍可能被负载顶开。喂食采用 50 μs、线材 100 μs、方块 20 μs 积分步长；当前软指尖会使抓持和多相机录制低于实时速度。头部峰值为本地刚性碰撞指标，不是人体安全标定。

## 查看相机画面

`.venv/bin/python scripts/preview_cameras.py` 生成 `outputs/previews/camera_views.png`：前两排为之前本地四视角，最后一排为当前 ViSk / FTF 共用的两台固定相机。场景和机械臂姿态一致；顶部与腕部相机仅用于生成历史布局对照，不加入当前采集场景。
