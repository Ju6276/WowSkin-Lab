"""Render the previous local four-view rig and current shared two-view rig."""
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
from wowskin_lab.data.cameras import add_camera
from wowskin_lab.envs.pick_cube import PandaPickCubeTactileEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/previews/camera_views.png'))
    args = parser.parse_args()
    env = PandaPickCubeTactileEnv(pollen=False)
    assert env.model.ncam == 2 and (env.model.cam_bodyid == 0).all()
    # Add only the two historical local observation views for this preview.
    # No extra cameras are added to the actual recording environment.
    xml = ET.fromstring(env.xml)
    add_camera(xml.find('worldbody'), 'previous_overhead', [.10, -.35, .50], [.4, 0, .14])
    hand = xml.find(".//body[@name='hand']")
    z = float(env.model.site_pos[env.tcp_site, 2])
    add_camera(hand, 'previous_wrist', [.10, 0., .075], [0., 0., z+.08], 90)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    data = mujoco.MjData(model)
    data.qpos[:] = env.data.qpos
    data.qvel[:] = env.data.qvel
    mujoco.mj_forward(model, data)
    images = {}
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for name in ('lab_front', 'lab_side', 'previous_overhead', 'previous_wrist'):
            renderer.update_scene(data, camera=name)
            images[name] = renderer.render().copy()
    fig, axes = plt.subplots(3, 2, figsize=(12, 13), facecolor='white')
    entries = [('lab_front', 'Previous ViSk / pixels1: fixed view A'),
               ('lab_side', 'Previous ViSk / pixels2: fixed view B'),
               ('previous_overhead', 'Previous ViSk / pixels51: overhead'),
               ('previous_wrist', 'Previous ViSk / pixels52: wrist'),
               ('lab_front', 'Now ViSk + FTF / pixels1: fixed view A'),
               ('lab_side', 'Now ViSk + FTF / pixels2: fixed view B')]
    for ax, (name, title) in zip(axes.flat, entries):
        ax.imshow(images[name]); ax.set_title(title, fontsize=12); ax.axis('off')
    fig.suptitle('Previous local 4-view rig vs current shared 2-view rig\nSame MuJoCo scene and robot state; not paper camera extrinsics', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140); plt.close(fig)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
