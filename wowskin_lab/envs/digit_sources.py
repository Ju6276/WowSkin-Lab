"""Import the two authors' distinct DIGIT mechanical models without remeshing.

Only instance names, collision bit routing and the rigid Panda mount change.
Effective defaults are materialized from an isolated source compilation so
Panda/AnySkin XML defaults cannot alter the sensor. The TACTO example includes
its long stub; it is deliberately retained, not replaced with a local housing.
"""
import copy
from functools import lru_cache
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from wowskin_lab.paths import ROOT

TACTO = ROOT/'third_party/mujoco-tacto/examples/xml'
FOTS = ROOT/'third_party/FOTS-mujoco'
CONTACT_FIELDS = ('friction', 'solref', 'solimp', 'condim', 'margin', 'gap', 'priority', 'solmix')


def numbers(value):
    return ' '.join(map(str, np.atleast_1d(value)))


@lru_cache(maxsize=2)
def source(sensor):
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    if sensor == 'tacto':
        common = ET.parse(TACTO/'tacto_common.xml').getroot()
        root.append(copy.deepcopy(common.find('default')))
        assets = copy.deepcopy(common.find('asset'))
        for mesh in list(assets.findall('mesh')):
            if mesh.get('name') not in ('finger_0', 'finger_1'):
                assets.remove(mesh)
            else:
                mesh.set('file', str(TACTO/mesh.get('file')))
        original = ET.parse(TACTO/'touch_playground_rot.xml').find(".//body[@name='left_finger_0']")
        body = copy.deepcopy(original)
        # This is the test fixture's world pose and gravity compensation, not
        # an intrinsic sensor parameter. The robot mount supplies its pose.
        body.attrib = {'name': 'source_digit'}
        for item in body.iter('body'):
            item.attrib.pop('gravcomp', None)
    elif sensor == 'digit_fots':
        assets = copy.deepcopy(ET.parse(FOTS/'assets/gel/shared.xml').find('asset'))
        for mesh in list(assets.findall('mesh')):
            if not mesh.get('name').startswith('digit_'):
                assets.remove(mesh)
            else:
                mesh.set('file', str(FOTS/'assets/stls/gel'/mesh.get('file')))
        body = copy.deepcopy(ET.parse(FOTS/'assets/gel/digit_sensor.xml').find('body'))
        body.set('name', 'source_digit')
    else:
        raise ValueError(sensor)
    root.append(assets)
    ET.SubElement(root, 'worldbody').append(body)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return root, model, data


@lru_cache(maxsize=2)
def geometry(sensor):
    from wowskin_lab.core.optical_tactile import geom_mesh
    _, model, data = source(sensor)
    if sensor == 'tacto':
        # Source gel + four pad boxes face +Y. Keep the complete long stub.
        rotation = np.diag([1., -1., -1.])
        surface = np.array([-.014, -.0095, -.064])
    else:
        # The FOTS CAD's rounded distal end is -source X, not +source Y.
        # Map it to +finger Z (away from the palm), while +source Z, the
        # sensing normal, points into the gap along -finger Y. Move the
        # camera, gel, housing and collision layer together as one assembly.
        rotation = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
        surface = data.geom_xpos[model.geom('digit:friction').id].copy()
        surface[2] += model.geom_size[model.geom('digit:friction').id, 2]
    vertices = []
    for i in range(model.ngeom):
        mesh = geom_mesh(model, i)
        world = np.einsum('ij,kj->ik', mesh.vertices, data.geom_xmat[i].reshape(3, 3))+data.geom_xpos[i]
        vertices.append(np.einsum('ij,kj->ik', world, rotation))
    vertices = np.concatenate(vertices)
    transformed_surface = rotation@surface
    position = np.array([-transformed_surface[0], -transformed_surface[1], -vertices[:, 2].min()])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return dict(position=position, quaternion=quat, proximal_z=0.,
                gel_center_z=float((transformed_surface+position)[2]),
                source_surface=surface, rotation=rotation,
                bounds=vertices+position)


def add(asset, parent, name, *, sensor, friction=None):
    root, model, _ = source(sensor)
    prefix = 'native_'+sensor+'_'
    for original in root.find('asset'):
        item = copy.deepcopy(original)
        item.set('name', prefix+item.get('name'))
        if asset.find(f"{item.tag}[@name='{item.get('name')}']") is None:
            asset.append(item)
    module = copy.deepcopy(root.find('worldbody/body'))
    old_bodies = list(module.iter('body'))
    for b in old_bodies:
        old_name = b.get('name')
        bid = model.body(old_name).id
        b.set('name', name+'_module' if b is module else name+'_'+old_name)
        b.attrib.pop('childclass', None)
        # Explicit source mass/COM/inertia, including the FOTS mesh-derived
        # values. No local bounding-box inertia or fake calibration.
        previous = b.find('inertial')
        if previous is not None:
            b.remove(previous)
        if model.body_mass[bid] > 0:
            ET.SubElement(b, 'inertial', mass=str(model.body_mass[bid]),
                          pos=numbers(model.body_ipos[bid]), quat=numbers(model.body_iquat[bid]),
                          diaginertia=numbers(model.body_inertia[bid]))
        for index, geom in enumerate(b.findall('geom')):
            # Unnamed visual geoms have no stable id: source body order does.
            gid = int(model.body_geomadr[bid])+index
            geom.set('name', name+'_g'+str(gid))
            geom.attrib.pop('class', None)
            for field in CONTACT_FIELDS:
                geom.set(field, numbers(getattr(model, 'geom_'+field)[gid]))
            # Resolve all inherited shape/render attributes from source.
            geom.set('type', ('plane','hfield','sphere','capsule','ellipsoid','cylinder','box','mesh')[int(model.geom_type[gid])])
            if geom.get('type') != 'mesh':
                geom.set('size', numbers(model.geom_size[gid]))
                geom.set('pos', numbers(model.geom_pos[gid]))
            for field in ('group',):
                geom.set(field, numbers(getattr(model, 'geom_'+field)[gid]))
            for field in ('mesh', 'material'):
                if geom.get(field):
                    geom.set(field, prefix+geom.get(field))
            active = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
            geom.set('contype', '2' if active else '0')
            geom.set('conaffinity', '7' if active else '0')
        for site in b.findall('site'):
            site.set('name', name+'_surface' if sensor == 'digit_fots' else name+'_source_site')
            # Sites are reference frames, not objects for RGB cameras.
            site.set('rgba', '0 0 0 0')
        for camera in b.findall('camera'):
            camera.set('name', 'optical_'+name+'_camera')
    transform = geometry(sensor)
    module.set('pos', numbers(transform['position']))
    module.set('quat', numbers(transform['quaternion']))
    for attr in ('euler', 'axisangle', 'xyaxes', 'zaxis'):
        module.attrib.pop(attr, None)
    parent.append(module)
    if sensor == 'tacto':
        # Register the author's configured flat optical gel (22 mm from its
        # camera) to the mechanical pad front. This is a rigid frame mapping;
        # do not override Renderer._generate_gel_trimesh with a local CAD gel.
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(ROOT/'third_party/mujoco-tacto/tacto/cfg/config_digit.yml')
        distance = -float(cfg.sensor.gel.origin[2])
        ET.SubElement(parent, 'site', name=name+'_camera',
                      pos=numbers([0., distance, transform['gel_center_z']]),
                      quat='.7071067811865476 -.7071067811865476 0 0',
                      size='.001', rgba='0 0 0 0')


def registration(sensor, name):
    _, model, _ = source(sensor)
    if sensor == 'digit_fots':
        return ('optical_'+name+'_camera', name+'_g'+str(model.geom('digit:friction').id))
    ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body('left_finger_pad').id
           and (model.geom_contype[i] or model.geom_conaffinity[i])]
    return name+'_camera', tuple(name+'_g'+str(i) for i in ids)
