from pathlib import Path
import json
import math
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hdm_grad_metrics import with_grad_metrics
from swanlab_hdm import Publisher, metric_fields


class Sink:
    def __init__(self):
        self.rows = []

    def log(self, data, *, step):
        self.rows.append((step, data))


def example_grid(root, step=1000):
    path = root / 'samples' / f'step-{step:06d}'
    path.mkdir(parents=True)
    rows = []
    for i in range(1, 7):
        name = f'{i:02}.png'
        Image.new('RGB', (16, 16), (30*i, 25, 50)).save(path / name)
        rows.append({'file': name, 'caption': 'private caption'})
    (path / 'manifest.json').write_text(json.dumps({'step': step, 'rows': rows}), encoding='utf-8')


class PublisherTest(TestCase):
    def test_live_rows_and_delayed_preview_are_not_duplicated(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sink = Sink()
            publisher = Publisher(root, sink, lambda img, caption: (img.size, caption))
            log = root / 'metrics.jsonl'
            log.write_text('{"step":1000,"loss":0.7}\n{"step":1010,', encoding='utf-8')
            publisher.flush()
            self.assertEqual([step for step, _ in sink.rows], [1000])
            (root / 'grad-metrics.jsonl').write_text('{"step":1010,"grad_norm":null}\n', encoding='utf-8')
            example_grid(root)
            with log.open('a', encoding='utf-8') as f:
                f.write('"loss":NaN,"lr":[0.1,0.2]}\n')
            publisher.flush()
            publisher.flush()
            self.assertEqual([step for step, _ in sink.rows], [1000, 1000, 1010, 1010])
            self.assertEqual(sink.rows[1][1], {'samples/grid': ((1184, 792), 'Optimizer step 1000')})
            self.assertEqual(sink.rows[2][1], {'health/grad_norm_nonfinite': 1})
            self.assertEqual(sink.rows[3][1]['health/loss_nonfinite'], 1)
            self.assertNotIn('train/loss', sink.rows[3][1])

    def test_after_step_and_scalars_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            example_grid(root, 2000)
            (root / 'metrics.jsonl').write_text('{"step":1000,"loss":0.7}\n{"step":1010,"loss":0.6}\n', encoding='utf-8')
            sink = Sink()
            Publisher(root, sink, None, after_step=1000, images=False).flush()
            self.assertEqual(sink.rows, [(1010, {'train/loss': 0.6, 'health/loss_nonfinite': 0})])

    def test_scalar_allowlist(self):
        self.assertEqual(metric_fields({'secret': 'never publish', 'loss': math.inf, 'lr': [1, 2]}),
                         {'health/loss_nonfinite': 1, 'train/lr_xut': 1, 'train/lr_connector': 2})
        self.assertEqual(metric_fields({'lr': [1, 2, 3]}),
                         {'train/lr_xut_cmuon': 1, 'train/lr_xut_adamw': 2, 'train/lr_connector_adamw': 3})


class Base:
    def __init__(self, cfg):
        self.output = cfg['output_dir']
        self.steps = []

    def on_before_optimizer_step(self, *args):
        self.steps.append('before')

    def on_train_batch_end(self, *args):
        self.steps.append('end')


class GradMetricsTest(TestCase):
    def test_cadence_only_publishes_at_optimizer_boundary(self):
        utils = ModuleType('torch.nn.utils')
        utils.get_total_norm = lambda grads, norm_type: SimpleNamespace(item=lambda: math.hypot(*grads))
        optimizer = SimpleNamespace(param_groups=[{'params': [
            SimpleNamespace(grad=SimpleNamespace(detach=lambda k=k: k)) for k in (3, 4)]}])
        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch.nn.utils': utils}):
            path = Path(tmp) / 'grad-metrics.jsonl'
            cb = with_grad_metrics(Base)({'output_dir': path.parent, 'grad_norm_every': 10})
            trainer = SimpleNamespace(global_step=9, is_global_zero=False)
            cb.on_before_optimizer_step(trainer, None, optimizer)
            self.assertIsNone(cb.pending_norm)
            trainer.is_global_zero = True
            cb.on_before_optimizer_step(trainer, None, optimizer)
            cb.on_train_batch_end(trainer, None, None, None, 0)
            self.assertFalse(path.exists())
            trainer.global_step = 10
            cb.on_train_batch_end(trainer, None, None, None, 0)
            cb.on_train_batch_end(trainer, None, None, None, 1)
            self.assertEqual(path.read_text(encoding='utf-8').splitlines(), ['{"step": 10, "grad_norm": 5.0, "grad_nonfinite": 0}'])
            self.assertEqual(cb.steps, ['before', 'before', 'end', 'end', 'end'])
