# AnySkin CAD 归类与 WowSkin 尺寸对照

五个 STL 已按安装平台归类，保留原始内容；与此前下载完全相同的副本经 SHA-256 核对后去重。来源：[AnySkin 官网](https://any-skin.github.io/) → [官方 CAD 文件夹](https://drive.google.com/drive/folders/1JOb_r0cT0t0BLju4XC6zboPppiXv8LDE)。

| 文件 | 分类 | 单件 X × Y × Z，mm | 单件数 |
| --- | --- | --- | --- |
| [OnRobot.stl](anyskin/onrobot/OnRobot.stl) | OnRobot 指尖安装结构 | **43.588 × 19.000 × 42.282** | 1 |
| [Franka_pair.stl](anyskin/franka/Franka_pair.stl) | Franka 成对指尖 | 66.088 × 20.000 × 42.282 | 2 |
| [xArm_pair.stl](anyskin/xarm/xArm_pair.stl) | xArm 成对指尖 | 68.088 × 18.000 × 42.282 | 2 |
| [xArm_pair_thin.stl](anyskin/xarm/xArm_pair_thin.stl) | xArm 较薄成对指尖 | 68.088 × 15.000 × 42.282 | 2 |
| [LEAP.stl](anyskin/leap/LEAP.stl) | LEAP 手指安装结构 | 48.578 × 29.074 × 42.282 | 1 |
| WowSkin 官网图 | 图示组件包络 | **43.600 × 11.400 × 42.200** | — |

**OnRobot 单件的外部包络最接近。** X、Z 分别差约 −0.012 mm、+0.082 mm，Y 厚度多 7.6 mm。平面尺寸相近支持将其作为几何参考，但不能证明孔位、内部空腔、皮肤和总厚度完全相同。

用户现已说明所持组件是打印底座，上面另套硅胶：底座约 **11.5 mm**，装好总高约 **13.5 mm**，硅胶只包住上沿。这与将官网 11.4 mm 当成带皮肤总高的旧假设不同。原始 `OnRobot.stl` 的 19 mm 是整个安装件的包络厚度，不能当成用户底座厚度，更不能用它与产品图的差额推算皮肤厚度。

测量方法：读取二进制 STL 顶点，按连通分量分离成对指尖，计算各轴 max−min。此批模型 X/Z 是主要轮廓方向，Y 为厚度/安装方向。STL 不含单位元数据；将原坐标按米解释后换算为毫米，与产品尺度一致。成对模型的整体尺寸包含两件间距，不能当作单件厚度。

STL 不标注五个磁力计的位置或磁化分布。芯片位置查 PCB，见 [五点布局](../../docs/MODEL_CALIBRATION_ZH.md#五个磁力计的位置)。

[对照图](comparison.png) · [尺寸 JSON](dimensions.json) · [来源及哈希](manifest.json)。项目根目录运行 `.venv/bin/python -m wowskin_lab.tools.inspect_cad` 可复核。

当前软体模型使用 [OnRobot 投影轮廓](onrobot_outline.json)作为参考，并按暂定 WowSkin 平面尺寸缩放；原始 STL 的整体厚度、孔位和背部安装结构没有直接移植。原始 CAD 的 MuJoCo 渲染仅用来对照，不是软体传感器模型。

独立传感器与 Franka 使用同一 [sensor_hardware.json](../../configs/sensor_hardware.json) 覆盖旧参数：底座 11.5 mm、顶部增加 2 mm、总高 13.5 mm。短侧边暂设下包 2 mm、外包络每边增加 0.5 mm，均待测；本模型用按参考轮廓挤出的底座，不包含原 STL 的孔位和内部结构。硅胶顶部和侧边共用三维弹性网格，与底座接触的内表面节点固定，不模拟套件脱落或相对滑动。Franka 已同步该尺寸及上沿硅胶套；PandaPickCube 使用较粗的任务网格，共享外包络尺寸但并非相同网格精度。
