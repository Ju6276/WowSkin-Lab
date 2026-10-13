"""Shared magnetic readout metadata and validity contract for recorders."""
import numpy as np

def magnetic_baseline(env):
    """Match the baseline to the selected observation backend, in its channel order."""
    response = getattr(env, 'reskin_response', None)
    return (np.asarray(response.model.meta['baseline_ut']) if response
            else env.baseline.reshape(15))


def magnetic_metadata(env):
    response = getattr(env, 'reskin_response', None)
    if not response:
        profile = getattr(env, 'magnetic_fit', None)
        return dict(magnetic_source='fitted_dipole' if profile else 'unfitted_dipole',
                    magnetic_fit_profile=profile, real_transfer_validated=False,
                    simulation_proxy_only=True,
                    mesh_quality_policy='0 < min V/V0 <= 0.15: warning only, live computed field and recording continue; inversion/nonfinite stops physics')
    return dict(magnetic_source='3dcal_local_mlp', channel_mapping=response.profile,
                contact_model='centroid XY + maximum normal depth; NOT full deformation; unvalidated',
                missing_tactile_policy='unavailable; never fill zero or use another backend',
                simulation_proxy_only=True, real_transfer_validated=False,
                magnetic_model_metadata=response.model.meta)


class MissingTactileError(ValueError):
    """The observation backend cannot supply this sample; physics may be valid."""


def require_training_tactile(env):
    """Reject missing observations, while allowing explicitly labelled previews."""
    if getattr(env, 'is_optical', False):
        if not np.isfinite(env.data.qpos).all() or np.any(env.data.warning.number):
            raise MissingTactileError('Invalid optical physics state')
        return  # Images validated on their actual camera clock by OpticalWriter.
    values = np.asarray(env.observation()['magnetic_ut'])
    response = getattr(env, 'reskin_response', None)
    row = getattr(env, 'last_row', None) or {}
    if row.get('tactile_valid') is False:
        raise MissingTactileError('Tactile observation invalid: '+row.get('tactile_status', 'unknown')+
                                 '. Held display values are not training measurements.')
    query_valid = not response or np.isfinite(row.get('reskin_query_mm', [np.nan]*3)).all()
    if values.shape != (15,) or not np.isfinite(values).all() or not query_valid:
        raise MissingTactileError('Tactile observation unavailable: this network has no valid '
                         'unloaded/recovery or current-contact prediction. Cannot write '
                         'a complete training sample; no zero filling or backend substitution.')
