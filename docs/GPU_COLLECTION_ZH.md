# GPU 采集已撤下

采集留在 CPU MuJoCo，Mac 和 Linux 使用同一条物理路径。先前可选的 `--physics-backend warp`、`--camera-backend warp` 和 `scripts/setup_capture.py --gpu` 已从入口删除。

在 RTX 4060 上，单台 Franka 的 CUDA Graph 路径比同机 CPU 更慢，完整光学录制也没有达到实时。测量和未通过项留在 [4060 对照记录](GPU_VALIDATION_ZH.md)，那份记录描述的是已删除的代码，不能再按其中的命令运行。
