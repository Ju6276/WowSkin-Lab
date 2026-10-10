"""OnRobot projected outline reference with provisional manufacturer dimensions.

No exact WowSkin CAD is implied. Skin contour and internal layer split remain
approximations. The same outline drives rendering AND the deformable volume.
"""
import json
from wowskin_lab.paths import ROOT
import numpy as np


def standard_mesh(width, length, bottom, top, outline_samples=48):
    # Use the official OnRobot projected contour as a geometric reference.
    # It is not proof that WowSkin's skin and mounting part are identical.
    reference = json.loads((ROOT/'assets/cad/onrobot_outline.json').read_text())
    hull = np.asarray(reference['vertices_mm'])
    closed = np.vstack([hull, hull[0]])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    sample_at = np.linspace(0, distance[-1], outline_samples, endpoint=False)
    outline = np.c_[np.interp(sample_at, distance, closed[:, 0]),
                    np.interp(sample_at, distance, closed[:, 1])]
    low, high = outline.min(axis=0), outline.max(axis=0)
    outline = ((outline-low)/(high-low)-.5)*[width, length]
    # Rings share angular indexing. All prisms use a globally consistent
    # diagonal so neighbouring tetrahedra have conforming shared faces.
    center = outline.mean(axis=0)
    xy = np.vstack([center, center+.5*(outline-center), outline])
    n = len(outline)
    triangles = []
    for i in range(n):
        j = (i+1) % n
        a, b, c, d = 1+i, 1+j, 1+n+i, 1+n+j
        triangles.extend([[0, a, b], [a, c, d], [a, d, b]])
    size = len(xy)
    points = np.vstack([np.c_[xy, np.full(size, bottom)],
                        np.c_[xy, np.full(size, top)]])
    elements = []
    for tri in triangles:
        a, b, c = sorted(tri)
        elements.extend([[a, b, c, c+size], [a, b, b+size, c+size],
                         [a, a+size, b+size, c+size]])
    # Area for an explicitly assumed 1000 kg/m^3 mass density.
    area = .5*abs(np.sum(outline[:, 0]*np.roll(outline[:, 1], -1)
                          - outline[:, 1]*np.roll(outline[:, 0], -1)))
    return outline, points, np.asarray(elements), size, area


def numbers(values):
    return " ".join(format(float(x), ".10g") for x in np.asarray(values).reshape(-1))


def cap_mesh(cfg):
    """Top layer plus short volumetric skirt; base-facing nodes are fixed.

    Skirt dimensions are explicit provisional parameters, not STL measurements.
    The inner boundary follows the same contour as the printed base.
    """
    base, top = cfg['base_height_m'], cfg['total_height_m']
    wall, depth = cfg['sleeve_wall_m'], cfg['sleeve_overlap_m']
    if not (0 < wall < min(cfg['width_m'], cfg['length_m'])/10 and 0 < depth < base):
        raise ValueError('Sleeve wall and overlap must be positive and smaller than the base')
    outline, _, _, _, _ = standard_mesh(cfg['width_m'], cfg['length_m'], base, top,
                                        cfg.get('mesh_outline_samples', 48))
    n = len(outline)
    outer = outline * (1 + 2*wall/np.array([cfg['width_m'], cfg['length_m']]))
    center = outline.mean(axis=0)
    xy = np.vstack([center, center+.5*(outline-center), outline, outer])
    count = len(xy)
    top_xy = xy.copy()
    # Redistribute an interior ring on the flat top without changing its
    # external contour, bottom attachment, thickness, or material volume.
    # A ring only 0.5 mm from the rim makes very thin top tetrahedra.
    fraction = cfg.get('cap_top_inner_fraction', 1.)
    if not .5 < fraction <= 1.:
        raise ValueError('cap_top_inner_fraction must be between 0.5 and 1')
    top_xy[1+n:1+2*n] = center+fraction*(outline-center)
    points = np.vstack([np.c_[xy, np.full(count, base)],
                        np.c_[top_xy, np.full(count, top)],
                        np.c_[outline, np.full(n, base-depth)],
                        np.c_[outer, np.full(n, base-depth)]])
    elements = []

    def prism(lower, upper):
        pairs = sorted(zip(lower, upper))
        (a, A), (b, B), (c, C) = pairs
        elements.extend([[a,b,c,C], [a,b,B,C], [a,A,B,C]])

    for i in range(n):
        j = (i+1) % n
        a,b,c,d,e,f = 1+i,1+j,1+n+i,1+n+j,1+2*n+i,1+2*n+j
        for tri in ([0,a,b], [a,c,d], [a,d,b], [c,e,f], [c,f,d]):
            prism(tri, [k+count for k in tri])
        lower_map = {c:2*count+i, d:2*count+j, e:2*count+n+i, f:2*count+n+j}
        for tri in ([c,e,f], [c,f,d]):
            # Match the upper ring's global ordering to the top slab triangulation.
            prism(tri, [lower_map[k] for k in tri])
    elements = np.asarray(elements)
    vertices = points[elements]
    volume = np.abs(np.linalg.det(np.stack([vertices[:,k]-vertices[:,0]
                                           for k in (1,2,3)], axis=-1))).sum()/6
    pins = list(range(1+2*n)) + list(range(2*count, 2*count+n))
    return outline, points, elements, pins, volume


def standard_xml(cfg):
    top = cfg["total_height_m"]
    bottom = top-cfg["skin_thickness_m"]
    outline, points, elements, pinned, area = standard_mesh(
        cfg["width_m"], cfg["length_m"], bottom, top, cfg.get("mesh_outline_samples", 48))
    mass = area*cfg['skin_thickness_m']*1000
    pin = f'<pin range="0 {pinned-1}"/>'
    shell_top = bottom-.00015
    if 'base_height_m' in cfg:
        if not np.isclose(cfg['base_height_m'], bottom, atol=1e-10, rtol=0):
            raise ValueError('Base height + top skin thickness must equal total height')
        shell_top = cfg['base_height_m']
    if cfg.get('sleeve_overlap_m', 0):
        outline, points, elements, pins, volume = cap_mesh(cfg)
        pin = '<pin id="'+' '.join(map(str,pins))+'"/>'
        mass = volume*1000
    shell = np.vstack([np.c_[outline, np.zeros(len(outline))],
                       np.c_[outline, np.full(len(outline), shell_top)]])
    asset = f'<mesh name="standard_shell" vertex="{numbers(shell)}"/>'
    housing = '''<geom name="standard_shell" type="mesh" mesh="standard_shell"
                     rgba=".055 .057 .062 1" contype="0" conaffinity="0"/>'''
    # MuJoCo's internal vertex/opposite-face contacts prevent tetrahedra from
    # folding through themselves. They are numerical constraints, not a new
    # material law; retain the deformation validity check in magnetic_field.
    internal = 'true' if cfg.get('fem_internal_contact', False) else 'false'
    radius = cfg.get('fem_contact_radius_m', .00005)
    timeconst = cfg.get('fem_contact_timeconst_s', .002)
    impedance = (f' solimp="{numbers(cfg["fem_contact_solimp"])}"'
                 if 'fem_contact_solimp' in cfg else '')
    flex = f'''<flexcomp name="skin" type="direct" dim="3"
                point="{numbers(points)}" element="{' '.join(map(str, elements.reshape(-1)))}"
                mass="{mass}" radius="{radius}"
                flatskin="true" rgba="{cfg.get('skin_rgba', '.32 .27 .25 1')}">
      {pin}
      <elasticity young="{cfg['young_pa']}" poisson="{cfg['poisson']}" damping=".00001"/>
      <contact selfcollide="none" internal="{internal}" condim="3"
               friction="{cfg['friction']} .005 .0001" solref="{timeconst} 1"{impedance}/>
    </flexcomp>'''
    return asset, housing, flex
