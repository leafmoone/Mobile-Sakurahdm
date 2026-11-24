#!/usr/bin/env python3
"""Publish HDM JSONL metrics and generated preview grids without touching training."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import time


class JsonlTail:
    def __init__(self, path):
        self.path = path
        self.position = 0

    def read(self):
        try:
            stream = self.path.open('rb')
        except FileNotFoundError:
            return
        with stream:
            stream.seek(self.position)
            while line := stream.readline():
                if not line.endswith(b'\n'):
                    break  # The writer has not committed the last row yet.
                self.position = stream.tell()
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    print(f'[swanlab] skipped malformed row in {self.path.name}', flush=True)
                    continue
                if isinstance(row, dict):
                    yield row


def metric_fields(row):
    result = {}
    for key in ('loss', 'grad_norm'):
        if key not in row:
            continue
        value = row[key]
        valid = type(value) in (int, float) and math.isfinite(value)
        if valid:
            result[f'train/{key}'] = value
        result[f'health/{key}_nonfinite'] = int(not valid)
    lr = row.get('lr')
    if isinstance(lr, list):
        labels = ('xut_cmuon', 'xut_adamw', 'connector_adamw') if len(lr) == 3 else ('xut', 'connector')
        for index, label in enumerate(labels):
            if index < len(lr) and type(lr[index]) in (int, float) and math.isfinite(lr[index]):
                result[f'train/lr_{label}'] = lr[index]
    return result


def preview_grid(directory):
    from PIL import Image, ImageOps

    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    rows = manifest['rows']
    if len(rows) != 6 or {row['file'] for row in rows} != {f'{i:02}.png' for i in range(1, 7)}:
        raise ValueError('Expected six published preview PNGs')
    size, gap = 384, 8
    grid = Image.new('RGB', (size * 3 + gap * 4, size * 2 + gap * 3), '#17212b')
    for i in range(6):
        with Image.open(directory / f'{i+1:02}.png') as image:
            tile = ImageOps.contain(image.convert('RGB'), (size, size), Image.Resampling.LANCZOS)
            x, y = gap + (size + gap) * (i % 3), gap + (size + gap) * (i // 3)
            grid.paste(tile, (x + (size - tile.width)//2, y + (size - tile.height)//2))
    return manifest['step'], grid


class Publisher:
    def __init__(self, directory, run, image_type, *, after_step=-1, images=True):
        self.directory = directory
        self.run = run
        self.image_type = image_type
        self.images = images
        self.last = {'metrics.jsonl': after_step, 'grad-metrics.jsonl': after_step, 'preview': after_step}
        self.tails = [JsonlTail(directory / f) for f in ('metrics.jsonl', 'grad-metrics.jsonl')]

    def flush(self):
        pending = []
        for tail in self.tails:
            key = tail.path.name
            for row in tail.read():
                step = row.get('step')
                if type(step) is int and step > self.last[key]:
                    pending.append((step, key, metric_fields(row)))
                    self.last[key] = step
        if self.images:
            for directory in (self.directory / 'samples').glob('step-*'):
                m = re.fullmatch(r'step-(\d+)', directory.name)
                if m and int(m[1]) > self.last['preview'] and (directory / 'manifest.json').is_file():
                    pending.append((int(m[1]), 'preview', directory))
        for step, key, data in sorted(pending, key=lambda row: (row[0], row[1])):
            if key == 'preview':
                self.last[key] = step
                try:
                    actual, grid = preview_grid(data)
                    if type(actual) is not int or actual != step:
                        raise ValueError('Preview step mismatch')
                except (OSError, TypeError, ValueError, KeyError) as error:
                    print(f'[swanlab] preview step={step} skipped: {type(error).__name__}', flush=True)
                    continue
                data = {'samples/grid': self.image_type(grid, caption=f'Optimizer step {step}')}
            if data:
                self.run.log(data, step=step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--project', default='Mobile-Sakurahdm')
    parser.add_argument('--workspace')
    parser.add_argument('--name', default='hdm-512')
    parser.add_argument('--mode', choices=('online', 'offline'), default='online')
    parser.add_argument('--once', action='store_true', help='Upload existing records and exit instead of following')
    parser.add_argument('--after-step', type=int, default=-1,
                        help='Only import steps later than this cutoff; always creates a new experiment')
    parser.add_argument('--scalars-only', action='store_true')
    parser.add_argument('--log-dir', type=Path)
    args = parser.parse_args()
    if args.once and not args.run_dir.is_dir():
        parser.error('run_dir does not exist')
    if args.mode == 'online':
        if not os.getenv('SWANLAB_API_KEY'):
            parser.error('Set SWANLAB_API_KEY in the environment (not as a CLI argument)')
        if not (os.getenv('https_proxy') or os.getenv('HTTPS_PROXY')):
            parser.error('Online mode requires the training-host HTTPS proxy')
        # This process only calls the tracking service, never SDS or other intranet endpoints.
        os.environ.pop('no_proxy', None)
        os.environ.pop('NO_PROXY', None)
    log_dir = args.log_dir or args.run_dir / 'swanlog'
    log_dir.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    run = None
    status = 0
    try:
        import swanlab
        settings = swanlab.Settings(interactive=False, terminal=swanlab.Settings.Terminal(proxy_type='none'),
            probe=swanlab.Settings.Probe(hardware=False, runtime=False, requirements=False, conda=False,
                                        git=False, monitor=False, swanlab=False))
        run = swanlab.init(project=args.project, workspace=args.workspace, experiment_name=args.name,
                           mode=args.mode, logdir=str(log_dir), public=False,
                           on_online_error='offline', settings=settings)
        print(f'[swanlab] mode={run.mode}, id={run.id}', flush=True)
        if run.mode == 'online':
            print(f'[swanlab] {run.url}', flush=True)
        publisher = Publisher(args.run_dir, run, swanlab.Image, after_step=args.after_step, images=not args.scalars_only)
        while not stopped:
            publisher.flush()
            if args.once:
                break
            time.sleep(2)
        if stopped:
            publisher.flush()
    except Exception as error:
        # Avoid printing arbitrary SDK exception messages that may include a URL or credentials.
        print(f'[swanlab] publisher stopped: {type(error).__name__}', flush=True)
        status = 1
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as error:
                print(f'[swanlab] finish failed: {type(error).__name__}', flush=True)
                status = 1
    return status


if __name__ == '__main__':
    raise SystemExit(main())
