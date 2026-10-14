"""Verify that restoring an experiment reproduces the next actual Adam update.

Run with the same isolated runtime as the training job. The input checkpoint is
read-only; this is a real learner check, not a continuation of the experiment.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wowskin_lab.learning.gelsight_ddqn import load_sources, make_learner, write_json
from wowskin_lab.learning.gelsight_ddqn_experiment import load_resume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import h5py
    with h5py.File(args.experiment/'resume.h5', 'r') as saved:
        report = json.loads(saved.attrs['report'])
    modules, manifest, tf = load_sources(args.workspace, args.runtime, report['seed'])
    import numpy as np
    if manifest != report['source']:
        raise ValueError('Source workspace differs from the checkpoint')
    parameters = SimpleNamespace(**report['learning'], **{'image_size': report['environment']['image_size']})
    learner = make_learner(modules, parameters)
    load_resume(args.experiment/'resume.h5', learner, report, tf)
    first = learner.learn()
    if first is None:
        raise ValueError('Checkpoint must contain more replay entries than the batch size')
    loss = float(first.history['loss'][0])
    expected = learner.q_evaluation.get_weights()
    expected_adam = tf.keras.backend.batch_get_value(learner.q_evaluation.optimizer.weights)
    next_random = np.random.rand(8)
    load_resume(args.experiment/'resume.h5', learner, report, tf)
    second_loss = float(learner.learn().history['loss'][0])
    np.testing.assert_allclose(second_loss, loss, atol=2e-5, rtol=2e-5)
    maximum = 0.
    for actual, wanted in zip(learner.q_evaluation.get_weights(), expected, strict=True):
        maximum = max(maximum, float(np.max(np.abs(actual-wanted))))
        np.testing.assert_allclose(actual, wanted, atol=2e-5, rtol=2e-5)
    for actual, wanted in zip(tf.keras.backend.batch_get_value(learner.q_evaluation.optimizer.weights), expected_adam, strict=True):
        np.testing.assert_allclose(actual, wanted, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(np.random.rand(8), next_random)
    result = dict(status='passed', checkpoint_step=report['completed_steps'],
                  next_loss=loss, restored_next_loss=second_loss, maximum_weight_difference=maximum,
                  replay_rng_matches=True, adam_next_update_matches=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
