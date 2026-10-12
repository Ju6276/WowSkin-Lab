"""Local care-room dressing and a fixed seated adult for the feeding task.

Procedural geometry, not an EBiM asset or a biomechanical human model. The
body/chair have rigid collision proxies; distant room decoration is visual.
The original head and its contact measurement remain in feeding.py.
"""
import xml.etree.ElementTree as ET

import numpy as np


def numbers(values):
    return ' '.join(f'{v:.8g}' for v in np.asarray(values).ravel())


def geom(parent, name, kind, pos, size, material, collision=False, **kwargs):
    return ET.SubElement(parent, 'geom', name=name, type=kind, pos=numbers(pos),
                         size=numbers(size), material=material, group='1', density='0',
                         contype='1' if collision else '0',
                         conaffinity='49' if collision else '0', **kwargs)


def limb(parent, name, start, end, radius, material, collision=True):
    return ET.SubElement(parent, 'geom', name=name, type='capsule',
                         fromto=numbers([start, end]), size=str(radius), material=material,
                         group='1', density='0', contype='1' if collision else '0',
                         conaffinity='49' if collision else '0')


def torso_mesh(asset):
    # Tailored rounded shirt silhouette: hips, waist, chest, shoulders, collar.
    sections = np.array([
        [-.555, .145, .095, .020], [-.535, .175, .110, .020],
        [-.47, .175, .110, .022], [-.38, .175, .110, .022],
        [-.27, .195, .112, .020], [-.19, .225, .098, .020],
        [-.145, .205, .090, .025], [-.115, .155, .082, .025],
        [-.090, .080, .051, .027], [-.078, .048, .043, .027]])
    from scipy.interpolate import PchipInterpolator
    z = np.linspace(sections[0, 0], sections[-1, 0], 35)
    ring = PchipInterpolator(sections[:, 0], sections[:, 1:])(z)
    theta = np.arange(48)*2*np.pi/48
    vertices = np.array([[rx*np.cos(t), cy+ry*np.sin(t), height]
                         for height, (rx, ry, cy) in zip(z, ring) for t in theta])
    faces = []
    for j in range(len(z)-1):
        for i in range(48):
            a, b = j*48+i, j*48+(i+1)%48
            faces.extend([[a, b, a+48], [b, b+48, a+48]])
    # Cap both ends. Explicit triangles avoid convexifying the visual waist.
    for i in range(1, 47):
        faces.extend([[0, i+1, i], [(len(z)-1)*48, (len(z)-1)*48+i,
                                   (len(z)-1)*48+i+1]])
    ET.SubElement(asset, 'mesh', name='care_shirt_mesh', smoothnormal='true', vertex=numbers(vertices),
                  face=' '.join(str(v) for face in faces for v in face))


def add_seated_adult(root, head_position, head_quat, skin_color):
    asset, world = root.find('asset'), root.find('worldbody')
    colors = dict(care_shirt='.24 .39 .46 1', care_collar='.79 .83 .80 1',
                  care_trousers='.19 .22 .27 1', care_shoes='.16 .12 .10 1',
                  care_chair_fabric='.42 .50 .39 1', care_chair_wood='.50 .31 .17 1',
                  care_skin=numbers(skin_color))
    for name, color in colors.items():
        ET.SubElement(asset, 'material', name=name, rgba=color, specular='.08', shininess='.12')
    torso_mesh(asset)
    body = ET.SubElement(world, 'body', name='care_seated_person', pos=numbers(head_position),
                         quat=numbers(head_quat))
    # Scalp-fitted hair is attached to the head in feeding.py.
    ET.SubElement(body, 'geom', name='care_shirt_visual', type='mesh', mesh='care_shirt_mesh',
                  material='care_shirt', group='1', density='0', contype='0', conaffinity='0')
    ET.SubElement(body, 'geom', name='care_torso_collision', type='ellipsoid',
                  pos='0 .02 -.36', size='.19 .10 .19', rgba='0 0 0 0', group='3',
                  density='0', contype='1', conaffinity='49')
    # Ring collar (not a solid disc crossing through the neck).
    for i in range(48):
        t, u = i*2*np.pi/48, (i+1)*2*np.pi/48
        limb(body, f'care_collar_{i}', [.048*np.cos(t), .027+.043*np.sin(t), -.078],
             [.048*np.cos(u), .027+.043*np.sin(u), -.078], .004, 'care_collar', False)
    geom(body, 'care_pelvis', 'ellipsoid', [0, .005, -.535], [.18, .12, .08], 'care_trousers', True)
    for side in (-1, 1):
        name = 'left' if side < 0 else 'right'
        # Sleeves, bare wrists and hands resting on the chair arms.
        shoulder = [side*.181, .020, -.185]
        elbow = [side*.253, -.005, -.400]
        wrist = [side*.253, -.201, -.475]
        limb(body, f'care_{name}_sleeve_upper', shoulder, elbow, .061, 'care_shirt')
        limb(body, f'care_{name}_sleeve_lower', elbow, wrist, .044, 'care_shirt')
        limb(body, f'care_{name}_wrist', [side*.253, -.195, -.475],
             [side*.253, -.228, -.475], .022, 'care_skin')
        geom(body, f'care_{name}_hand', 'ellipsoid', [side*.253, -.248, -.475],
             [.036, .043, .019], 'care_skin', True)
        for finger in range(4):
            x = side*.253+(finger-1.5)*.014
            length = (.055, .067, .064, .049)[finger if side > 0 else 3-finger]
            knuckle = [x, -.278-length*.56, -.477]
            limb(body, f'care_{name}_finger_{finger}_proximal', [x, -.278, -.474],
                 knuckle, .0075, 'care_skin', False)
            limb(body, f'care_{name}_finger_{finger}_distal', knuckle,
                 [x, -.278-length, -.491], .006, 'care_skin', False)
        limb(body, f'care_{name}_thumb', [side*.228, -.233, -.477],
             [side*.201, -.260, -.481], .010, 'care_skin', False)
        limb(body, f'care_{name}_thumb_tip', [side*.201, -.260, -.481],
             [side*.201, -.279, -.491], .008, 'care_skin', False)
        hip, knee, ankle = [side*.100, -.005, -.540], [side*.115, -.345, -.568], [side*.115, -.345, -.943]
        limb(body, f'care_{name}_thigh', hip, knee, .085, 'care_trousers')
        limb(body, f'care_{name}_shin', knee, ankle, .063, 'care_trousers')
        geom(body, f'care_{name}_shoe', 'ellipsoid', [side*.115, -.392, -.968],
             [.068, .125, .044], 'care_shoes', True)

    chair = ET.SubElement(world, 'body', name='care_chair', pos=numbers(head_position),
                          quat=numbers(head_quat))
    geom(chair, 'care_chair_seat', 'box', [0, -.065, -.632], [.275, .25, .035], 'care_chair_wood', True)
    geom(chair, 'care_chair_cushion', 'ellipsoid', [0, -.06, -.602], [.28, .26, .043], 'care_chair_fabric')
    geom(chair, 'care_chair_back', 'box', [0, .179, -.39], [.265, .035, .225], 'care_chair_wood', True)
    geom(chair, 'care_chair_back_cushion', 'ellipsoid', [0, .134, -.38], [.251, .050, .227], 'care_chair_fabric')
    for side in (-1, 1):
        geom(chair, f'care_armrest_{side}', 'box', [side*.264, -.055, -.527],
             [.032, .245, .020], 'care_chair_wood', True)
        for y in (-.22, .13):
            limb(chair, f'care_chair_leg_{side}_{y}', [side*.238, y, -.638],
                 [side*.26, y+.035, .349-head_position[2]+.027], .025, 'care_chair_wood')
    return body


def add_care_room(root):
    asset, world, visual = root.find('asset'), root.find('worldbody'), root.find('visual')
    colors = dict(care_wall='.84 .82 .75 1', care_wainscot='.65 .71 .65 1',
                  care_trim='.91 .89 .83 1', care_oak='.67 .48 .29 1',
                  care_floor='.54 .40 .27 1', care_frame='.29 .25 .20 1',
                  care_glass='.59 .76 .81 1', care_curtain='.75 .69 .56 1',
                  care_cabinet='.75 .70 .58 1', care_leaf='.22 .36 .21 1',
                  care_pot='.57 .32 .23 1', care_paper='.94 .91 .81 1', care_book_blue='.24 .39 .46 1')
    for name, color in colors.items():
        ET.SubElement(asset, 'material', name=name, rgba=color, specular='.08', shininess='.15')
    # The table contact sizes, positions and friction remain the EBiM values.
    for e in world.findall('geom'):
        if e.get('name', '').startswith('visual_box_Rectangle'):
            e.set('material', 'care_oak')
        if e.get('name') == 'reference_floor':
            e.attrib.pop('rgba', None); e.set('material', 'care_floor'); e.set('group', '1')
    room = ET.SubElement(world, 'body', name='care_room_decor')
    floor = .349
    # Warm floor with fine board seams, kept out of collision computation.
    for i in range(27):
        x = -4.0+i*.18
        geom(room, f'care_floor_seam_{i}', 'box', [x, -.45, floor+.0002],
             [.0008, 2.0, .00015], 'care_frame')
    geom(room, 'care_wall_back', 'box', [-1.50, 1.40, floor+1.40], [2.65, .045, 1.40], 'care_wall')
    geom(room, 'care_wall_side', 'box', [.45, -.6, floor+1.40], [.045, 2.0, 1.40], 'care_wall')
    geom(room, 'care_wall_lower', 'box', [-1.5, 1.346, floor+.44], [2.60, .014, .44], 'care_wainscot')
    geom(room, 'care_wall_rail', 'box', [-1.5, 1.324, floor+.89], [2.60, .018, .022], 'care_trim')
    geom(room, 'care_skirt', 'box', [-1.5, 1.32, floor+.045], [2.60, .016, .045], 'care_trim')
    # Daylit window on the far wall, away from both acquisition cameras.
    geom(room, 'care_window_frame', 'box', [-1.8, 1.328, 2.10], [.72, .045, .58], 'care_trim')
    geom(room, 'care_window_glass', 'box', [-1.8, 1.276, 2.10], [.67, .006, .53], 'care_glass')
    for dx in (-.68, 0, .68):
        geom(room, f'care_window_upright_{dx}', 'box', [-1.8+dx, 1.26, 2.10], [.018, .016, .55], 'care_trim')
    geom(room, 'care_window_crossbar', 'box', [-1.8, 1.26, 2.12], [.69, .018, .016], 'care_trim')
    geom(room, 'care_window_sill', 'box', [-1.8, 1.235, 1.54], [.78, .13, .022], 'care_oak')
    for side in (-1, 1):
        for i in range(6):
            geom(room, f'care_curtain_{side}_{i}', 'capsule', [-1.8+side*(.74+i*.035), 1.22, 2.02],
                 [.032, .69], 'care_curtain')
    # A low sideboard, books and an abstract framed print give scale/context.
    geom(room, 'care_sideboard', 'box', [-.45, 1.10, .76], [.39, .18, .32], 'care_cabinet')
    geom(room, 'care_sideboard_top', 'box', [-.45, 1.10, 1.097], [.41, .20, .017], 'care_oak')
    for x in (-.64, -.25):
        geom(room, f'care_cabinet_door_{x}', 'box', [x, .913, .77], [.183, .01, .286], 'care_trim')
        geom(room, f'care_cabinet_pull_{x}', 'box', [x, .896, .90], [.050, .008, .006], 'care_frame')
    for i, mat in enumerate(('care_book_blue', 'care_pot', 'care_wainscot')):
        geom(room, f'care_book_{i}', 'box', [-.57+i*.065, 1.08, 1.19], [.025, .09, .08+i*.008], mat)
    geom(room, 'care_art_frame', 'box', [-.37, 1.30, 1.85], [.30, .027, .35], 'care_frame')
    geom(room, 'care_art_paper', 'box', [-.37, 1.267, 1.85], [.28, .005, .33], 'care_paper')
    for i, (x, z, mat) in enumerate(((-.45, 1.93, 'care_wainscot'), (-.29, 1.77, 'care_pot'))):
        geom(room, f'care_art_shape_{i}', 'ellipsoid', [x, 1.258, z], [.115, .006, .135], mat)
    # Indoor plant: compact closed geoms, all visual-only.
    geom(room, 'care_plant_pot', 'cylinder', [-.08, .64, .53], [.14, .18], 'care_pot')
    for i in range(12):
        angle = i*2.4
        stem = [-.08, .64, .66]
        tip = [-.08+.20*np.cos(angle), .64+.17*np.sin(angle), .86+.038*i]
        limb(room, f'care_plant_stem_{i}', stem, tip, .005, 'care_leaf', False)
        geom(room, f'care_plant_leaf_{i}', 'ellipsoid', tip, [.065, .023, .14], 'care_leaf',
             euler=numbers([.35*np.sin(angle), .5*np.cos(angle), angle]))
    # Dedicated visual group for an architectural cutaway in the free viewer.
    # Acquisition cameras still render the complete room. No collision changes.
    for e in room.findall('geom'):
        if e.get('name').startswith(('care_wall_', 'care_window_', 'care_curtain_', 'care_art_', 'care_skirt')):
            e.set('group', '4')
    # Replace the warehouse lighting copied from EBiM with a local room rig.
    for e in list(world.findall('light')):
        world.remove(e)
    ET.SubElement(world, 'light', name='care_daylight', pos='-1.7 .9 2.8', dir='0 -.5 -1',
                  directional='true', diffuse='.72 .70 .65', specular='.15 .15 .15', castshadow='true')
    ET.SubElement(world, 'light', name='care_fill', pos='-2.5 -1.2 2.4', dir='.2 .4 -1',
                  directional='true', diffuse='.38 .40 .42', specular='.08 .08 .08', castshadow='false')
    visual.find('headlight').set('ambient', '.28 .28 .28')
    visual.find('headlight').set('diffuse', '.45 .45 .45')
    visual.find('headlight').set('specular', '.10 .10 .10')
    quality = visual.find('quality')
    if quality is not None:
        quality.set('shadowsize', '2048')
