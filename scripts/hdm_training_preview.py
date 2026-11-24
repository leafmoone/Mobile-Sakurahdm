"""Opt-in PNG previews for the pinned Lightning HDM trainer (no extra SDS reads)."""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import time

import torch


def with_training_previews(artifact_class):
    """Extend, rather than replace, the upstream checkpoint/metrics callback."""
    from sakurahdm.sample import sample

    class TrainingPreviews(artifact_class):
        def __init__(self, training):
            super().__init__(training)
            cfg = training.get('preview', {})
            self.every = cfg.get('every_steps', 0)
            self.count = cfg.get('count', 6)
            if type(self.every) is not int or self.every < 0 or type(self.count) is not int or not 1 <= self.count <= 32:
                raise ValueError('Invalid training.preview interval or image count')
            self.recent = deque(maxlen=max(128, self.count * 8))
            self.last_preview = 0

        def on_train_start(self, trainer, pl_module):
            super().on_train_start(trainer, pl_module)
            self.last_preview = int(trainer.global_step)

        def remember(self, batch):
            for row in batch:
                caption = row.get('prompt')
                if not isinstance(caption, str) or not caption.strip() or row.get('task') != 't2i':
                    continue
                self.recent.append(dict(caption=caption, sample_id=row.get('sample_id'),
                                        source_shard=row.get('source_shard'), key=row.get('key')))

        def select(self, step, seed):
            unique, seen = [], set()
            for row in reversed(self.recent):
                identity = str(row['sample_id']) if row['sample_id'] is not None else (row['source_shard'], row['key'])
                if identity == (None, None):
                    identity = row['caption']
                if identity not in seen:
                    unique.append(row)
                    seen.add(identity)
            if len(unique) < self.count:
                raise ValueError(f'Only {len(unique)} distinct nonempty T2I captions available, want {self.count}')
            return random.Random(f'hdm-preview:{seed}:{step}').sample(unique, self.count)

        def render(self, module, step):
            from xut import env as xut_env

            start = time.monotonic()
            destination = self.output / 'samples' / f'step-{step:06d}'
            if destination.exists():
                if (destination / 'manifest.json').is_file():
                    print(f'[train-preview] step={step} already published', flush=True)
                    return
                raise RuntimeError(f'Unexpected existing preview path: {destination}')
            config = deepcopy(module.config)
            seed = int(config['sample'].get('seed', config['training']['seed']))
            rows = self.select(step, seed)
            destination.parent.mkdir(parents=True, exist_ok=True)
            was_training = module.training
            was_compiled = xut_env.TORCH_COMPILE
            try:
                module.eval()
                # Offline sampler was verified in eager mode. Avoid specializing
                # the training compile cache for six inference-only shapes.
                xut_env.TORCH_COMPILE = False
                with torch.random.fork_rng(devices=[module.device]), torch.autocast('cuda', enabled=False), tempfile.TemporaryDirectory(
                    prefix='.preview-', dir=destination.parent
                ) as temp:
                    pending = Path(temp)
                    for index, row in enumerate(rows, 1):
                        row['seed'] = seed + index - 1
                        row['file'] = f'{index:02}.png'
                        config['sample']['seed'] = row['seed']
                        sample(module.model, module.processor, row['caption'], config=config).save(pending / row['file'])
                    manifest = dict(step=step, steps=config['sample']['steps'], cfg=config['sample']['guidance_scale'], rows=rows)
                    (pending / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    pending.rename(destination)
            finally:
                xut_env.TORCH_COMPILE = was_compiled
                module.train(was_training)
            print(f'[train-preview] step={step} images={len(rows)} seconds={time.monotonic()-start:.1f} path={destination}', flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            step = int(trainer.global_step)
            # At batch end, the final microbatch reports the incremented step.
            # Collect the other microbatches of each target update at step N-1.
            due = self.every and step > self.last_preview and step > 0 and step % self.every == 0
            if trainer.is_global_zero and self.every and (due or (step + 1) % self.every == 0):
                self.remember(batch)
            if not due:
                return
            self.last_preview = step
            trainer.strategy.barrier('training previews start')
            try:
                if trainer.is_global_zero:
                    self.render(pl_module, step)
            except Exception as error:
                # Preserve a successfully saved training checkpoint even if a
                # qualitative preview fails; make any failure visible in logs.
                print(f'[train-preview] step={step} FAILED: {type(error).__name__}: {error}', flush=True)
            finally:
                self.recent.clear()
                trainer.strategy.barrier('training previews complete')

    return TrainingPreviews
