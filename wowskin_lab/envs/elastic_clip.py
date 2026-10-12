"""Experimental C-clip as passive discrete bending beams in its YZ section.

The mounting base is fixed. Rectangular beam elements bend about local X;
their length and width are inextensible, so this is not a 3-D shell FEM.
Every moving collision surface moves with its visible beam. No grasp assist,
capture-zone force, state overwrite, weld, or success-dependent dynamics.
"""
import json
import xml.etree.ElementTree as ET
import numpy as np
from wowskin_lab.paths import ROOT


def configuration():
    return json.loads((ROOT/'configs/elastic_clip.json').read_text())


def _text(values):
    return ' '.join(f'{v:.12g}' for v in np.asarray(values).ravel())


def _base_visual(asset, height):
    """Keep the source STL below the attachment plane; visual only, no hull collision."""
    content = (ROOT/'assets/ebim/C-Clip.stl').read_bytes()
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3,3)), ('attribute', '<u2')])
    triangles = np.frombuffer(content, dtype=dtype, offset=84)['vertices'].astype(float)*.01
    faces = []
    for triangle in triangles:
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            inside_a, inside_b = a[2] <= height, b[2] <= height
            if inside_a:
                polygon.append(a)
            if inside_a != inside_b:
                polygon.append(a+(b-a)*(height-a[2])/(b[2]-a[2]))
        for i in range(1, len(polygon)-1):
            faces.append([polygon[0], polygon[i], polygon[i+1]])
    vertices = np.asarray(faces).reshape(-1, 3)
    ET.SubElement(asset, 'mesh', name='elastic_clip_mount_visual',
                  vertex=_text(vertices), face=_text(np.arange(len(vertices)).reshape(-1, 3)))


def _beam_chain(body, c, contact, midline, span_thickness, prefix, tip_name):
    """Only the first hinge attaches to the fixed installation body."""
    E, width = float(c['young_pa']), float(c['width_m'])
    if len(span_thickness) != len(midline)-1 or min(span_thickness) <= 0:
        raise ValueError('One positive thickness is required per clip span')
    points, thickness = [np.asarray(midline[0])], []
    for a, b, t in zip(midline, midline[1:], span_thickness):
        a, b = np.asarray(a), np.asarray(b)
        n = int(np.ceil(np.linalg.norm(b-a)/c['max_segment_length_m']))
        if n == 0:
            raise ValueError('Clip spans must have nonzero length')
        points.extend(a+(b-a)*i/n for i in range(1, n+1))
        thickness.extend([t]*n)
    points = np.array(points)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    inertia = width*np.asarray(thickness)**3/12
    parent = body
    for i, (start, end, length, t) in enumerate(zip(points, points[1:], lengths, thickness)):
        position = [c['x_center_m'], *start] if i == 0 else [0, *(start-points[i-1])]
        segment = ET.SubElement(parent, 'body', name=f'{prefix}segment_{i}', pos=_text(position))
        # Compliance of the two half elements around an internal hinge.
        compliance = length/(2*E*inertia[i])
        if i:
            compliance += lengths[i-1]/(2*E*inertia[i-1])
        ET.SubElement(segment, 'joint', name=f'{prefix}bend_{i}', type='hinge',
                      axis='1 0 0', limited='false', stiffness=str(1/compliance),
                      springref='0', damping=str(c['joint_damping_nm_s_rad']), armature='0', frictionloss='0')
        delta = end-start
        angle = np.arctan2(delta[1], delta[0])
        ET.SubElement(segment, 'geom', name=f'{prefix}surface_{i}', type='box',
                      pos=_text([0, *(delta/2)]), size=_text([width/2, length/2, t/2]),
                      quat=_text([np.cos(angle/2), np.sin(angle/2), 0, 0]), **contact)
        if i == len(lengths)-1:
            ET.SubElement(segment, 'site', name=tip_name, pos=_text([0, *delta]),
                          size='.0001', rgba='0 0 0 0')
        parent = segment


def add_elastic_clip(root, body, cfg):
    """Replace the shell, preserving fixed installation and passive contact."""
    c = cfg['elastic_clip']
    if c['young_pa'] <= 0 or c['density_kg_m3'] <= 0 or c['max_segment_length_m'] <= 0:
        raise ValueError('Elastic clip material and discretization must be positive')
    for g in list(body.findall('geom')):
        if g.get('name') != 'cclip_0_base':
            body.remove(g)
    _base_visual(root.find('asset'), c['base_cut_height_m'])
    ET.SubElement(body, 'geom', name='elastic_clip_mount_visual', type='mesh',
                  mesh='elastic_clip_mount_visual', contype='0', conaffinity='0',
                  group='1', mass='0', rgba='1 .85 .05 1')
    contact = dict(contype='1', conaffinity='7', condim='3',
                   friction=_text(c['friction']), solref=_text(c['contact_solref']),
                   solimp='.95 .99 .001', margin='0', rgba='1 .85 .05 1',
                   density=str(c['density_kg_m3']))
    _beam_chain(body, c, contact, c['midline_yz_m'], c['span_thickness_m'],
                'elastic_clip_', 'elastic_clip_upper_tip')
    if 'lower_midline_yz_m' in c:
        _beam_chain(body, c, contact, c['lower_midline_yz_m'], c['lower_span_thickness_m'],
                    'elastic_clip_lower_', 'elastic_clip_lower_tip')
    else:  # Reconstruct explicitly recorded v1 experiments without changing them.
        lower = [c['x_center_m'], *c['fixed_lower_lip_center_yz_m']]
        ET.SubElement(body, 'geom', name='elastic_clip_lower_lip', type='box',
                      pos=_text(lower), size=_text([c['width_m']/2, *c['fixed_lower_lip_half_yz_m']]), **contact)
        ET.SubElement(body, 'site', name='elastic_clip_lower_tip',
                      pos=_text([lower[0],lower[1],lower[2]+c['fixed_lower_lip_half_yz_m'][1]]),
                      size='.0001', rgba='0 0 0 0')


def diagnostics(model, data):
    upper = data.site_xpos[model.site('elastic_clip_upper_tip').id]
    lower = data.site_xpos[model.site('elastic_clip_lower_tip').id]
    ids = [i for i in range(model.njnt) if (model.joint(i).name or '').startswith(('elastic_clip_bend_', 'elastic_clip_lower_bend_'))]
    lower_ids = [i for i in ids if model.joint(i).name.startswith('elastic_clip_lower_bend_')]
    q = data.qpos[model.jnt_qposadr[ids]]
    return dict(clip_tip_distance_m=float(np.linalg.norm(upper-lower)),
                clip_elastic_energy_j=float(.5*np.sum(model.jnt_stiffness[ids]*q*q)),
                clip_max_bend_rad=float(np.max(np.abs(q))),
                clip_lower_max_bend_rad=float(np.max(np.abs(data.qpos[model.jnt_qposadr[lower_ids]]))) if lower_ids else 0.,
                clip_upper_tip_m=upper.copy(), clip_lower_tip_m=lower.copy())
