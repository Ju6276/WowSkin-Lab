"""Prevent duplicate metrics, local log writers and checkpoint off-by-one errors."""
import unittest

from wowskin_lab.learning.training_runtime import CompletedUpdateCheckpoints, WandbLogger


class FakeRun:
    def __init__(self):
        self.records = []

    def log(self, values, step):
        self.records.append((step, values))


class TrainingRuntimeChecks(unittest.TestCase):
    def test_upstream_duplicate_loss_is_one_wandb_record_per_update(self):
        run = FakeRun()
        logger = WandbLogger(run)
        logger.log_metrics({'actor_loss': 2.5}, 0, 'train')
        with logger.log_and_dump_ctx(0, 'train') as log:
            log('actor_loss', 2.5)
            log('total_time', 0.3)
        logger.log_metrics({'actor_loss': 1.5}, 1, 'train')
        logger.close()
        logger.close()
        self.assertEqual([s for s, _ in run.records], [1, 2])
        self.assertEqual(run.records[0][1], {'train/actor_loss': 2.5,
            'train/total_time': 0.3, 'train/completed_updates': 1})
        self.assertEqual(run.records[1][1]['train/actor_loss'], 1.5)
        self.assertFalse(hasattr(logger, '_sw'))
        self.assertFalse(hasattr(logger, '_train_mg'))

    def test_checkpoint_contents_match_completed_updates_and_no_zero_file(self):
        class Workspace:
            global_step = 0
            optimizer_updates = 0
            snapshots = []

            def save_snapshot(self):
                self.snapshots.append((self.global_step, self.optimizer_updates))

        ws = Workspace()
        schedule = CompletedUpdateCheckpoints(ws, 3000)
        while ws.global_step < 10000:
            schedule.before_update(ws.global_step)
            ws.optimizer_updates += 1  # Original agent.update occurs here.
            ws.global_step += 1        # Original loop increments AFTER update.
        schedule.finish()
        schedule.finish()
        self.assertEqual(ws.snapshots, [(3000, 3000), (6000, 6000),
                                        (9000, 9000), (10000, 10000)])

    def test_final_checkpoint_on_interval_is_not_duplicated(self):
        class Workspace:
            global_step = 3
            saved = []

            def save_snapshot(self):
                self.saved.append(self.global_step)

        ws = Workspace()
        schedule = CompletedUpdateCheckpoints(ws, 3)
        schedule.before_update(3)
        schedule.finish()
        self.assertEqual(ws.saved, [3])


if __name__ == '__main__':
    unittest.main()
