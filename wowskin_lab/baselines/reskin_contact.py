"""Experimental single-probe surrogate for MuJoCo contact, not a calibrated skin.

Uses actual FEM top-surface compression (never a guessed force-to-depth law).
A cube patch is represented by its force-weighted contact centroid and maximum
top-surface compression. The dataset does NOT validate this patch reduction.
Strict mode therefore refuses a cube prediction; explicit preview extrapolation
can inspect its failure modes without treating the output as validated data.
"""
import mujoco
import numpy as np
from .reskin_forward import ResponseModel


class ReskinContactResponse:
    def __init__(self, model_path, extrapolate=False):
        self.model = ResponseModel(model_path)
        self.profile = self.model.meta.get('mapping')
        if not self.profile:
            raise ValueError('MuJoCo requires a model trained with an explicit mapping profile')
        self.extrapolate = bool(extrapolate)
        self.q = np.asarray(self.profile['dataset_xy_to_board'], dtype=float)
        if self.q.shape != (2, 2) or not np.allclose(self.q.T@self.q, np.eye(2)):
            raise ValueError('Dataset/board mapping must be an orthogonal 2D transform')
        self.origin = np.asarray(self.profile['dataset_origin_mm'])

    def query(self, sim):
        m, d = sim.model, sim.data
        rotation = d.xmat[sim.mount_body].reshape(3, 3) if sim.mount_body is not None else np.eye(3)
        origin = d.xpos[sim.mount_body] if sim.mount_body is not None else np.zeros(3)
        positions, weights = [], []
        for i in range(d.ncon):
            contact = d.contact[i]
            contact_geoms = getattr(sim, 'contact_geoms', {sim.probe_geom})
            if not any(int(g) in contact_geoms for g in contact.geom) or sim.skin_flex not in contact.flex:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] > 1e-8:
                positions.append((contact.pos-origin)@rotation)
                weights.append(force[0])
        if not weights:
            return None, 0., 0
        positions = np.array(positions)
        centroid = np.average(positions, axis=0, weights=weights)
        top = np.isclose(sim.rest[:, 2], sim.rest[:, 2].max(), atol=1e-9)
        compression = np.maximum(sim.rest[top, 2]-sim.skin_vertices_local()[top, 2], 0.)
        depth_mm = float(compression.max()*1000)
        xy_mm = self.q.T@(centroid[:2]*1000)+self.origin
        span_mm = float(np.linalg.norm(np.ptp(positions[:, :2], axis=0))*1000)
        return np.r_[xy_mm, depth_mm], span_mm, len(weights)

    def evaluate(self, sim):
        query, span, contacts = self.query(sim)
        common = dict(magnetic_source='3dcal_local_mlp_provisional_mapping',
                      magnetic_channel_order='candidate_center_left_right_up_down',
                      magnetic_axes='source_ReSkin_XYZ_unverified',
                      reskin_patch_span_mm=span, reskin_contact_count=contacts,
                      reskin_transfer_validated=False)
        if query is None:
            # A reference convention, NOT a model trained on unloaded examples.
            return dict(common, magnetic_ut=np.zeros(15), ideal_delta_ut=np.zeros(15),
                        raw_field_ut=np.asarray(self.model.meta['baseline_ut']),
                        reskin_query_mm=np.full(3, np.nan), reskin_status='unloaded_reference_zero',
                        reskin_in_training_domain=False, reskin_prediction_emitted=False)
        issues = self.model.domain_issues(query)
        within_domain = not any(s != 'unvalidated_depth_interpolation' for s in issues)
        if getattr(sim, 'contact_geometry', '') == 'cable':
            issues.append('cable_line_contact_not_training_probe')
        elif m_is_box(sim):
            issues.append('cube_patch_not_training_probe')
        else:
            issues.append('probe_geometry_transfer_unverified')
        # No clipping of XY/depth, no invented shallow-contact scaling, no sum
        # over collision points (which would make amplitude mesh-dependent).
        emitted = self.extrapolate
        values = (self.model.predict(query, allow_extrapolation=True) if emitted else np.full(15, np.nan))
        status = ('EXPERIMENTAL:' if emitted else 'BLOCKED:')+';'.join(issues)
        return dict(common, magnetic_ut=values, ideal_delta_ut=values.copy(),
                    raw_field_ut=values+np.asarray(self.model.meta['baseline_ut']),
                    reskin_query_mm=query, reskin_status=status,
                    reskin_in_training_domain=within_domain, reskin_prediction_emitted=emitted)


def m_is_box(sim):
    return sim.model.geom_type[sim.probe_geom] == mujoco.mjtGeom.mjGEOM_BOX
