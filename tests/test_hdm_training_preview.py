"""CPU-only tests for the callback's step boundary and artifact ordering."""
import importlib.util
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch


def preview_class():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'hdm_training_preview.py'
    spec = importlib.util.spec_from_file_location('preview_test_module', path)
    mod = importlib.util.module_from_spec(spec)
    mock_sampler = ModuleType('sakurahdm.sample')
    mock_sampler.sample = lambda *args, **kwargs: None
    with patch.dict(sys.modules, {'torch': ModuleType('torch'),
                                  'sakurahdm': ModuleType('sakurahdm'),
                                  'sakurahdm.sample': mock_sampler}):
        spec.loader.exec_module(mod)
        return mod.with_training_previews(Artifacts)


class Artifacts:
    def __init__(self, training):
        self.output = Path(training['output_dir'])
        self.events = []

    def on_train_start(self, trainer, pl_module):
        self.events.append(('start', trainer.global_step))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.events.append(('artifacts', trainer.global_step))


class TrainingPreviewTest(TestCase):
    def callback(self):
        callback = preview_class()({'output_dir': 'unused', 'preview': {'every_steps': 1000, 'count': 6}})
        trainer = SimpleNamespace(global_step=0, is_global_zero=True)
        trainer.strategy = SimpleNamespace(barrier=lambda name: callback.events.append(('barrier', name)))
        return callback, trainer

    @staticmethod
    def rows():
        return [dict(task='t2i', prompt=f'caption {i}', sample_id=i,
                     source_shard='source/examples.tar', key=str(i)) for i in range(8)]

    def batch(self, callback, trainer, step, rows=None):
        trainer.global_step = step
        callback.on_train_batch_end(trainer, None, None, self.rows() if rows is None else rows, 0)

    def test_schedule_and_checkpoint_before_preview(self):
        callback, trainer = self.callback()
        callback.on_train_start(trainer, None)
        rng_state = random.getstate()
        def render(module, step):
            selected = callback.select(step, 42)
            self.assertEqual(len({row['sample_id'] for row in selected}), 6)
            self.assertEqual(selected, callback.select(step, 42))
            callback.events.append(('render', step))
        callback.render = render
        self.batch(callback, trainer, 998)
        self.assertEqual(len(callback.recent), 0)
        self.batch(callback, trainer, 999)
        self.batch(callback, trainer, 1000)
        self.assertEqual(callback.events[-4:], [('artifacts', 1000), ('barrier', 'training previews start'),
                                                ('render', 1000), ('barrier', 'training previews complete')])
        self.assertEqual(len(callback.recent), 0)
        self.batch(callback, trainer, 1000)
        self.assertEqual(callback.events.count(('render', 1000)), 1)
        self.assertEqual(random.getstate(), rng_state)

    def test_nonzero_rank_joins_barriers_without_sampling(self):
        callback, trainer = self.callback()
        trainer.is_global_zero = False
        callback.render = lambda *_: self.fail('Nonzero rank rendered preview')
        self.batch(callback, trainer, 999)
        self.batch(callback, trainer, 1000)
        self.assertEqual(callback.events[-3:], [('artifacts', 1000), ('barrier', 'training previews start'),
                                                ('barrier', 'training previews complete')])
        self.assertEqual(len(callback.recent), 0)

    def test_resuming_at_preview_boundary_does_not_repeat_it(self):
        callback, trainer = self.callback()
        trainer.global_step = 1000
        callback.on_train_start(trainer, None)
        callback.render = lambda *_: self.fail('Resume repeated preview')
        self.batch(callback, trainer, 1000)
        self.assertEqual(callback.events[-2:], [('start', 1000), ('artifacts', 1000)])


if __name__ == '__main__':
    main()
