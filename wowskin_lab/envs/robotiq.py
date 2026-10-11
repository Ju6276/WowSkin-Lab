"""EBiM's pinned 2F-85 mechanism, with local tactile-pad mounting on Panda."""
import math
import copy
import xml.etree.ElementTree as ET
from wowskin_lab.paths import ROOT

PREFIX = 'left_fr3v2_1_robotiq_85_'
DRIVER = PREFIX+'left_knuckle_joint'


def closure_for_gap(thickness, gap):
    # Source knuckle->finger->tip translations, cancelling tip hinge rotation.
    lo, hi = 0., .8
    for _ in range(50):
        a = (lo+hi)/2
        width = 2*(.0306011+.03715754*math.cos(a)-.04342173*math.sin(a)-.02525865-thickness)
        if width > gap: lo = a
        else: hi = a
    return (lo+hi)/2


def initialize_gap(model, data, thickness, gap):
    """Set RESET joint positions using the source mechanism equalities.

    This is an initial condition, not an attachment or runtime grasp helper.
    The sourced actuator, range and force limits control every subsequent step.
    """
    import mujoco
    import numpy as np
    angle = closure_for_gap(thickness, gap)
    driver = model.joint(DRIVER).id
    data.qpos[model.jnt_qposadr[driver]] = angle
    for eq in range(model.neq):
        if model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[eq] == driver:
            joint = model.eq_obj1id[eq]
            data.qpos[model.jnt_qposadr[joint]] = np.polynomial.polynomial.polyval(
                angle, model.eq_data[eq, :5])
    mujoco.mj_forward(model, data)
    return angle


def replace_gripper(xml, cfg):
    root = ET.fromstring(xml)
    source = ET.parse(ROOT/'assets/ebim/source_scene.xml').getroot()
    base = copy.deepcopy(source.find(f".//body[@name='{PREFIX}base_link']"))
    # Neutral defaults prevent Panda slide-joint/actuator defaults leaking into
    # the source hinge mechanism. Explicit source joint dynamics are retained.
    default = ET.SubElement(root.find('default'), 'default', {'class':'ebim_robotiq'})
    ET.SubElement(default, 'joint', type='hinge', limited='true')
    ET.SubElement(default, 'geom', condim='3', solref='0.004 1', solimp='0.95 0.99 0.001', margin='0.0005')
    base.set('childclass', 'ebim_robotiq')
    old = root.find(".//body[@name='hand']")
    base.set('name', 'hand')
    base.set('pos', old.get('pos'))
    base.set('quat', old.get('quat'))
    # Rotate source X jaw separation into Panda's Y jaw separation.
    mechanism = ET.Element('body', name='robotiq_mechanism', quat='.7071067811865476 0 0 .7071067811865476', childclass='ebim_robotiq')
    for e in list(base):
        base.remove(e)
        if e.tag != 'camera': mechanism.append(e)
    base.append(mechanism)
    for site in old.findall('site'):
        if site.get('name') != 'tactile_tcp': base.append(copy.deepcopy(site))
    # Keep the configured acquisition wrist camera when replacing the hand.
    # The upstream Robotiq preview camera is not one of our dataset views.
    for camera in old.findall('camera'):
        base.append(copy.deepcopy(camera))
    for side, mountname in [('left','wowskin_mount'),('right','silicone_mount')]:
        tip = mechanism.find(f".//body[@name='{PREFIX}{side}_finger_tip_link']")
        # EBiM routing pads/lips are replaced; original metal links are kept.
        for g in list(tip.findall('geom')):
            if g.get('class') == 'routing_pad': tip.remove(g)
        mount = copy.deepcopy(old.find(f".//body[@name='{mountname}']"))
        mount.set('pos', ('-.02525865 0 .030' if side == 'left' else '.02525865 0 .030'))
        # sensor X/Y lie in the fingertip plane; +Z faces the opposite pad.
        mount.set('quat', '.5 .5 -.5 -.5' if side == 'left' else '.5 .5 .5 .5')
        tip.append(mount)
    # Define a reference TCP at undeformed pad contact, independently of the
    # servo endpoint. The source driver still commands its full 0.8 rad range.
    contact_angle = closure_for_gap(cfg['total_height_m'], 0.)
    tcp_z = .0549045+.03715754*math.sin(contact_angle)+.04342173*math.cos(contact_angle)+.030
    ET.SubElement(base,'site',name='tactile_tcp',pos=f'0 0 {tcp_z}',size='.001',rgba='.1 .9 .7 .4')
    for parent in root.iter():
        if old in list(parent): parent.remove(old);parent.append(base);break
    # Replace only Panda finger transmission/constraints, leaving arm intact.
    for section in ('actuator','tendon','equality','contact'):
        node=root.find(section)
        if node is None: node=ET.SubElement(root,section)
        for e in list(node):
            if any('finger' in v or v == 'hand' for child in e.iter() for v in child.attrib.values()): node.remove(e)
        src=source.find(section)
        if src is not None:
            for e in src:
                refs=[v for k,v in e.attrib.items() if k in ('joint','joint1','joint2','body1','body2')]
                if refs and all(v.startswith(PREFIX) for v in refs):
                    item=copy.deepcopy(e)
                    if section=='actuator': item.set('name','fingers_actuator')
                    node.append(item)
    meshes={g.get('mesh') for g in base.iter('geom') if g.get('mesh','').startswith('robot_ee_robotiq')}
    materials={g.get('material') for g in base.iter('geom') if g.get('material','').startswith('mat_gripper')}
    for a in source.find('asset'):
        if a.get('name') in meshes|materials:
            a=copy.deepcopy(a)
            if a.tag=='mesh': a.set('file',str(ROOT/'assets/ebim/robotiq'/a.get('file')))
            if a.tag == 'material':
                # Local visual approximation of black 2F-85 hardware, using
                # Robotiq's product photography; not a measured reflectance.
                a.set('rgba', '0.035 0.038 0.042 1')
                a.set('specular', '0.25')
                a.set('shininess', '0.3')
            root.find('asset').append(a)
    # Source collision masks target the upstream world. Remap to this scene's
    # object/table groups; same mesh geometry, not a replacement collision shape.
    for g in base.iter('geom'):
        if g.get('mesh','').startswith('robot_ee_robotiq') and g.get('group')=='3':
            g.set('contype','8');g.set('conaffinity','5')
            # Collision geometry should also stay dark if explicitly shown.
            g.set('rgba', '0.035 0.038 0.042 1')
    # Retain the source mechanism's explicit exclusions only. Do not exclude
    # every robot link against both new mounts: that would hide cross-finger
    # collisions involving the rigid housings.
    cfg.update(gripper='robotiq_2f85', robotiq_closed_angle_rad=.8, robotiq_empty_gap_m=None, gripper_control_mode='ebim_robotiq_position',
               gripper_source='assets/ebim/robotiq/manifest.json',
               gripper_mount_status='Local mount at source fingertip mesh inner bounds, z=30 mm; 13.5 mm skin, not a manufactured adapter',
               gripper_force_status='EBiM kp=300 kv=20, hinge actuatorfrcrange +/-50; torque, NOT contact force in N')
    return ET.tostring(root, encoding='unicode')
