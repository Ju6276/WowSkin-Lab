"""Logging and checkpoint scheduling only; no policy or optimizer changes."""
from contextlib import contextmanager


class WandbLogger:
    """Original ViSk Logger interface, with W&B as the only metrics sink.

    The original loop logs a loss twice on console-log steps. Buffer one update
    so both calls share one W&B record. No TensorBoard/CSV writers are created.
    """

    def __init__(self, run):
        self.run = run
        self._step = None
        self._pending = {}

    def log(self, key, value, step):
        step = int(step)
        if self._step is not None and step < self._step:
            raise ValueError('Training log steps must not go backwards')
        if self._step != step:
            self._flush()
            self._step = step
        self._pending[key] = float(value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def _flush(self):
        if self._pending:
            self.run.log({**self._pending, 'train/completed_updates': self._step + 1},
                         step=self._step + 1)
            self._pending.clear()

    def dump(self, step, ty=None):
        # Wait for the next update (or close) to merge all metrics for this step.
        pass

    @contextmanager
    def log_and_dump_ctx(self, step, ty):
        yield lambda key, value: self.log(f'{ty}/{key}', value, step)

    def close(self):
        self._flush()


class CompletedUpdateCheckpoints:
    """Call upstream save_snapshot at actual completed-update boundaries.

    Before update N+1, WorkspaceIL.global_step is N and its parameters contain
    exactly N optimizer updates. Disable its old zero-based in-loop schedule
    with cfg.suite.save_every_steps=None, then call this before each update and
    finish() after train(). Snapshot format and global_step are unmodified.
    """

    def __init__(self, workspace, interval):
        if interval < 1:
            raise ValueError('Checkpoint interval must be positive')
        self.workspace = workspace
        self.interval = interval
        self.saved_steps = []

    def _save(self, step):
        if step > 0 and step not in self.saved_steps:
            if self.workspace.global_step != step:
                raise ValueError('Checkpoint step differs from completed updates')
            self.workspace.save_snapshot()
            self.saved_steps.append(step)

    def before_update(self, step):
        if step % self.interval == 0:
            self._save(step)

    def finish(self):
        self._save(self.workspace.global_step)
