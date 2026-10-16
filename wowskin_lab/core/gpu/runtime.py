"""Original MuJoCo models on CUDA, with reusable CUDA Graph capture.

No material replacement, silently discarded plugin forces or CPU fallback.
The optional cable adapter translates the native passive force law explicitly.
This is a physics utility, not the collection/observation environment.
"""
from contextlib import ExitStack
from unittest.mock import patch
import importlib.metadata
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp


class UnsupportedModel(ValueError):
    pass


def check_model(model, *, cable_plugin_warp=False):
    if cable_plugin_warp:
        from .cable import parameters
        parameters(model)
        return
    if model.nplugin or np.any(model.body_plugin>=0):
        raise UnsupportedModel('MuJoCo Warp 3.15 does not directly support body plugins. Use native CPU MuJoCo, or explicitly enable the local cable_plugin_warp force adapter; never silently drop cable elasticity.')


def capture_step(model,data,step,epa_horizon=128):
    """Keep upstream flex scratch outside conditional Graph bodies.

    The installed package is unchanged. Captured kernels and equations match
    MuJoCo Warp 3.15.0; this utility requires serial replay of its graph.
    EPA horizon is scratch capacity, not a contact-tolerance/material change.
    """
    if importlib.metadata.version('mujoco-warp')!='3.15.0':
        raise RuntimeError('CUDA Graph scratch adapter is verified for mujoco-warp==3.15.0 only')
    from mujoco_warp._src import passive,collision_convex,collision_flex
    scratch=()
    with ExitStack() as stack:
        if model.nflexvert:
            wvec=wp.empty((data.nworld,model.nflexvert),dtype=wp.vec3)
            wres=wp.empty_like(wvec);scratch=(wvec,wres)
            def reused(m,d,res,vec,s1_in=1.,s2_in=0.,use_timestep=True,non_simple_only=False,snh_only=False,skip=None):
                if m is not model:raise RuntimeError('Scratch belongs to one model')
                passive.flex_gather(m,d,wvec,vec,skip=skip)
                passive.flex_hessian_mul(m,d,wres,wvec,s1_in=s1_in,s2_in=s2_in,use_timestep=use_timestep,non_simple_only=non_simple_only,snh_only=snh_only,skip=skip)
                passive.flex_scatter(m,d,res,wres,scale=1.,skip=skip)
            stack.enter_context(patch.object(passive,'flex_stretch_mul',reused))
        stack.enter_context(patch.object(collision_convex,'MJ_MAX_EPAHORIZON',epa_horizon))
        stack.enter_context(patch.object(collision_flex,'MJ_MAX_EPAHORIZON',epa_horizon))
        with wp.ScopedCapture() as captured:step()
    return captured.graph,scratch


@wp.kernel
def joint_pd(qids: wp.array(dtype=int),vids: wp.array(dtype=int),aids: wp.array(dtype=int),target: wp.array(dtype=float),limits: wp.array2d(dtype=float),
             q: wp.array2d(dtype=float),v: wp.array2d(dtype=float),bias: wp.array2d(dtype=float),ctrl: wp.array2d(dtype=float)):
    i=wp.tid();a=aids[i];torque=600.*(target[i]-q[0,qids[i]])-45.*v[0,vids[i]]+bias[0,vids[i]]
    ctrl[0,a]=wp.clamp(torque,limits[a,0],limits[a,1])


class GpuSimulation:
    """Single-world GPU state. The caller supplies the original model and state.

    For task use, update joint/control targets between graph replays. Robot IK,
    images, tactile observations and recording remain the caller's responsibility.
    """
    def __init__(self,model,data,*,arm=None,nconmax=32768,njmax=16384,epa_horizon=128,cable_plugin_warp=False):
        check_model(model,cable_plugin_warp=cable_plugin_warp)
        wp.init()
        if not wp.is_cuda_available():raise RuntimeError('NVIDIA CUDA is required; no CPU fallback')
        wp.set_device('cuda')
        self.native_model=model;self.initial_data=data;self.capacity=(nconmax,njmax);self.epa_horizon=epa_horizon
        self.cable=None
        if cable_plugin_warp:
            from .cable import CableForces
            self.cable=CableForces(model)
        self.model=self.cable.import_model(model) if self.cable else mjw.put_model(model)
        self.data=mjw.put_data(model,data,nconmax=nconmax,njmax=njmax)
        if self.cable:self.cable.allocate(self.data,model.nbody)
        self.arm=None
        if arm:
            self.arm={k:wp.array(v,dtype=float if k=='target' else int) for k,v in arm.items()}
            self.limits=wp.array(model.actuator_ctrlrange,dtype=float)
        self.graph=None;self.scratch=()

    def raw_step(self):
        if self.arm:
            a=self.arm
            wp.launch(joint_pd,len(a['qids']),[a['qids'],a['vids'],a['aids'],a['target'],self.limits,self.data.qpos,self.data.qvel,self.data.qfrc_bias,self.data.ctrl])
        mjw.step(self.model,self.data)

    def reset(self):
        if self.graph is not None:raise RuntimeError('Reset before graph capture; a graph owns its array pointers')
        self.data=mjw.put_data(self.native_model,self.initial_data,nconmax=self.capacity[0],njmax=self.capacity[1])

    def compile(self):
        self.raw_step();wp.synchronize();self.reset()
        self.graph,self.scratch=capture_step(self.model,self.data,self.raw_step,self.epa_horizon)

    def step(self):
        if self.graph is None:raise RuntimeError('Call compile() before stepping')
        wp.capture_launch(self.graph)

    def diagnostics(self):
        flags=int(self.data.overflow.numpy()[0])
        names=[name for name,flag in mjw.OverflowType.__members__.items() if name not in ('NONE','ALL') and flags&int(flag)]
        q=self.data.qpos.numpy()[0];v=self.data.qvel.numpy()[0]
        return dict(overflow_flags=flags,overflow_names=names,
            finite=bool(np.isfinite(q).all() and np.isfinite(v).all()),solver_iterations=int(self.data.solver_niter.numpy()[0]),
            contacts=int(self.data.nacon.numpy()[0]),constraints=int(self.data.nefc.numpy()[0]))
