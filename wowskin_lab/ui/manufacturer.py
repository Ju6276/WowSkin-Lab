"""Execute the unmodified manufacturer's drawing function on an offscreen surface.

Only the nested draw function and its two layout assignments are loaded; the
upstream serial reader and window/event loop are not executed here. MIT source
snapshot, original background and SHA-256 provenance are in third_party/.
"""
import ast
import os
from functools import lru_cache
import numpy as np
from wowskin_lab.paths import ROOT
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

SOURCE = ROOT/'third_party/wowskin_visualizer'

@lru_cache(maxsize=1)
def upstream():
    path = SOURCE/'wowskin_viz.py'
    tree = ast.parse(path.read_text())
    outer = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='visualize')
    draw = next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='visualize_data')
    assignments = [n for n in outer.body if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id in ('chip_locations','chip_xy_rotations') for t in n.targets)]
    scope = dict(np=np, pygame=pygame)
    exec(compile(ast.Module(body=assignments+[draw],type_ignores=[]),str(path),'exec'),scope)
    defaults = dict(zip([a.arg for a in outer.args.args][-len(outer.args.defaults):],outer.args.defaults))
    scope['scaling'] = ast.literal_eval(defaults['scaling'])
    scope['viz_mode'] = '3axis'
    return scope

DEFAULT_SCALING = float(upstream()['scaling'])
DEFAULT_FULL_SCALE = 29*DEFAULT_SCALING  # compatibility with existing scale keys


def draw(surface, values, *, scaling=DEFAULT_SCALING, locations=None, rotations=None):
    # New globals per call: renderer instances cannot overwrite each other's axes.
    import types
    scope = dict(upstream())
    scope.update(window=surface, scaling=scaling)
    if locations is not None: scope['chip_locations'] = np.asarray(locations)
    if rotations is not None: scope['chip_xy_rotations'] = np.asarray(rotations)
    original = scope['visualize_data']
    function = types.FunctionType(original.__code__,scope,original.__name__)
    function(np.asarray(values,dtype=float))


def render(values, scaling=DEFAULT_SCALING):
    background = pygame.image.load(str(SOURCE/'wowskin_bg.png'))
    height = int(background.get_height()/background.get_width()*400)
    surface = pygame.Surface((400,height),pygame.SRCALPHA)
    surface.fill((234,237,232,255))
    surface.blit(pygame.transform.scale(background,(400,height)),(0,0))
    if values is not None: draw(surface,values,scaling=scaling)
    return np.transpose(pygame.surfarray.array3d(surface),(1,0,2)).copy()
