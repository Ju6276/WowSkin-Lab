"""Standalone sensor visualization and paired real/simulation capture."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "viewer"):
        p = commands.add_parser(name, help="Sensor-only press/release inspection")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--x-mm", type=float, default=0.)
        p.add_argument("--y-mm", type=float, default=0.)
        p.add_argument("--load-n", type=float, default=.05)
        p.add_argument("--ideal", action="store_true", help="Disable additive noise and drift")
        if name == "demo":
            p.add_argument("--output", default="outputs/sensor")
            p.add_argument("--render", action="store_true")
            p.set_defaults(func=lambda a: __import__("wowskin_lab.envs.bench", fromlist=["*"]).demo(a))
        else:
            p.add_argument("--seconds", type=float, default=0)
            p.add_argument("--real-port", help="Real WowSkin serial port; enables paired recording")
            p.add_argument("--real-baseline", help="Load a saved noise_summary.json baseline for the same device")
            p.add_argument("--capture-dir", default="outputs/calibration", help="Parent directory for a new real/sim recording")
            p.set_defaults(func=lambda a: __import__("wowskin_lab.envs.bench", fromlist=["*"]).viewer(a))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
