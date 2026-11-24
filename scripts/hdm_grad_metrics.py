"""Rank-zero gradient diagnostics, measured after AMP unscale and before clipping."""
import json
import math


def with_grad_metrics(base):
    class GradMetrics(base):
        def __init__(self, training):
            super().__init__(training)
            self.grad_interval = training.get('grad_norm_every', 0)
            if type(self.grad_interval) is not int or self.grad_interval < 0:
                raise ValueError('training.grad_norm_every must be a nonnegative integer')
            self.pending_norm = None

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            super().on_before_optimizer_step(trainer, pl_module, optimizer)
            # DDP accumulation has finished; Lightning's native FP16 AMP plugin
            # calls this hook after unscale for the non-fused optimizer we use.
            step = int(trainer.global_step) + 1
            if not (self.grad_interval and trainer.is_global_zero and step % self.grad_interval == 0):
                return
            from torch.nn.utils import get_total_norm

            grads = [p.grad.detach() for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            if grads:
                norm = float(get_total_norm(grads, norm_type=2.0).item())
                self.pending_norm = (step, norm)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.pending_norm and int(trainer.global_step) >= self.pending_norm[0]:
                step, norm = self.pending_norm
                self.pending_norm = None
                try:
                    self.output.mkdir(parents=True, exist_ok=True)
                    with (self.output / 'grad-metrics.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(dict(step=step, grad_norm=norm if math.isfinite(norm) else None,
                                                grad_nonfinite=int(not math.isfinite(norm))), allow_nan=False) + '\n')
                except OSError as error:
                    print(f'[grad-metrics] local write failed: {type(error).__name__}', flush=True)
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    return GradMetrics
