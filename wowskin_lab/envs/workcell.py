"""Industrial workcell dressing, built from native MuJoCo geometry/materials.

The existing tooling plate and cube remain the manipulation contact geometry.
Decorative markings, rails and background assets are non-colliding.
"""
import xml.etree.ElementTree as ET


def industrial_workcell(root):
    asset, world, visual = root.find('asset'), root.find('worldbody'), root.find('visual')
    def material(name, color, **kwargs):
        ET.SubElement(asset, 'material', name=name, rgba=color, **kwargs)
    material('cell_aluminium', '.48 .53 .57 1', specular='.45', shininess='.5')
    material('cell_graphite', '.11 .14 .17 1', specular='.22', shininess='.3')
    material('cell_plate', '.30 .35 .39 1', specular='.3', shininess='.35')
    material('cell_amber', '.91 .56 .12 1', specular='.1')
    material('cell_green', '.19 .58 .39 1', specular='.1')
    material('cell_wall', '.57 .61 .64 1', specular='.05')
    material('cell_panel', '.39 .45 .49 1', specular='.12')
    ET.SubElement(asset, 'texture', name='cell_sky', type='skybox', builtin='gradient',
                  rgb1='.43 .51 .59', rgb2='.78 .82 .85', width='512', height='3072')
    ET.SubElement(asset, 'texture', name='cell_floor_tex', type='2d', builtin='checker',
                  rgb1='.36 .40 .43', rgb2='.39 .43 .46', width='512', height='512')
    ET.SubElement(asset, 'material', name='cell_floor_mat', texture='cell_floor_tex',
                  texrepeat='8 8', texuniform='true', reflectance='.04', specular='.12')
    floor = world.find("geom[@name='floor']")
    floor.set('pos', '0 0 -.82')
    floor.set('size', '5 5 .1')
    floor.set('material', 'cell_floor_mat')
    floor.attrib.pop('rgba', None)
    head = visual.find('headlight')
    head.set('ambient', '.25 .25 .27')
    head.set('diffuse', '.45 .45 .45')
    head.set('specular', '.12 .12 .12')
    ET.SubElement(visual, 'rgba', haze='.66 .72 .77 1')
    for light in list(world.findall('light')):
        world.remove(light)
    ET.SubElement(world, 'light', name='cell_key', pos='.6 -.8 2.2', dir='-.2 .3 -1',
                  diffuse='.8 .78 .73', specular='.35 .35 .35', castshadow='true')
    ET.SubElement(world, 'light', name='cell_fill', pos='-.8 .6 1.7', dir='.4 -.2 -1',
                  diffuse='.4 .46 .52', castshadow='false')

    def geom(name, pos, size, mat='cell_aluminium', **attrs):
        def fmt(v): return ' '.join(str(x) for x in v)
        defaults = dict(type='box', pos=fmt(pos), size=fmt(size), material=mat,
                        contype='0', conaffinity='0', group='1')
        defaults.update(attrs)
        return ET.SubElement(world, 'geom', name='cell_'+name, **defaults)

    # Working deck at robot base height; original raised table is the fixture.
    geom('deck', [.30, 0, -.03], [.50, .40, .03], contype='1', conaffinity='7')
    geom('front_apron', [.30, -.395, -.095], [.50, .018, .045], 'cell_graphite')
    geom('rear_apron', [.30, .395, -.095], [.50, .018, .045], 'cell_graphite')
    for x in (-.15, .75):
        for y in (-.35, .35):
            tag=f'{x}_{y}'
            geom('leg_'+tag, [x,y,-.42], [.028,.028,.36])
            geom('slot_'+tag, [x+.0282,y,-.42], [.0005,.010,.33], 'cell_graphite')
            geom('foot_'+tag, [x,y,-.80], [.044,.044,.02], 'cell_graphite')
        geom('sidebrace_'+str(x), [x,0,-.65], [.022,.35,.022])
    for y in (-.35,.35):
        geom('crossbrace_'+str(y), [.30,y,-.65], [.45,.022,.022])
    geom('robot_base_plate', [0,0,.001], [.095,.10,.001], 'cell_graphite')
    table = world.find("geom[@name='table']")
    table.attrib.pop('rgba', None)
    table.set('material','cell_plate')
    # Plate mounting holes and corner fasteners, away from the manipulation zone.
    for x in (.235,.625):
        for y in (-.215,.215):
            geom(f'bolt_{x}_{y}',[x,y,.1003],[.005,.0005],'cell_graphite',type='cylinder')
    for x in (.235,.275,.315,.355,.395,.435,.475,.515,.555,.595,.625):
        for y in (-.21,.21):
            geom(f'hole_{x}_{y}',[x,y,.10015],[.0018,.00015],'cell_graphite',type='cylinder')
    # Flush visual guides: no artificial barriers or object attachment.
    for name,center,material_name in [('pick',(.40,0.),'cell_amber'),('place',(.49,.135),'cell_green')]:
        x,y=center
        for sign in (-1,1):
            geom(f'{name}_x{sign}',[x+sign*.040,y,.1002],[.0012,.0412,.0002],material_name)
            geom(f'{name}_y{sign}',[x,y+sign*.040,.1002],[.0412,.0012,.0002],material_name)
    # Rear guard frame and workshop equipment form the background, not obstacles.
    for y in (-.48,.48):
        geom('guard_post_'+str(y),[-.25,y,.32],[.02,.02,.46])
    geom('guard_top',[-.25,0,.78],[.02,.50,.02])
    geom('guard_panel',[-.251,0,.34],[.003,.455,.39],'cell_panel',rgba='.28 .42 .46 .18')
    geom('back_wall',[-1.05,0,.45],[.045,2.5,1.27],'cell_wall')
    geom('side_wall',[.4,1.35,.45],[1.5,.04,1.27],'cell_wall')
    for y in (-1.5,-.5,.5,1.5):
        geom('wall_seam_'+str(y),[-1.003,y,.45],[.001,.004,1.27],'cell_panel')
    geom('cabinet',[-.55,.74,-.25],[.20,.23,.56],'cell_graphite')
    geom('cabinet_door',[-.345,.74,-.25],[.006,.208,.52],'cell_panel')
    geom('cabinet_handle',[-.333,.58,-.2],[.013,.008,.10])
    geom('bin_shelf', [.18,.75,-.185], [.14,.23,.015])
    for y in (.55,.95):
        geom('shelf_leg_'+str(y), [.18,y,-.51], [.025,.025,.31])
    for y in (.61,.75,.89):
        geom('bin_'+str(y),[.18,y,-.08],[.10,.058,.09],'cell_panel')
        geom('bin_inset_'+str(y),[.18,y,.011],[.085,.046,.001],'cell_graphite')
    # Floor perimeter marking, sized to the workbench footprint.
    for y in (-.58,.58):
        geom('floor_line_'+str(y),[.30,y,-.8195],[.66,.012,.0005],'cell_amber')
    for x in (-.36,.96):
        geom('floor_end_'+str(x),[x,0,-.8195],[.012,.58,.0005],'cell_amber')
