#!/usr/bin/env python3
"""Explicit Mage-XUT launcher for an isolated fresh training or sampling run."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def install():
    from hdm_mage_xut import build_model
    from sakurahdm import models
    def select(config):
        if 'mage' not in config.get('vae', {}):
            raise ValueError('Mage-XUT launcher requires the Mage stage')
        return build_model(config)

    models.build_model = select
    return select


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'sample'))
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    sys.argv[1:] = args.arguments
    if args.mode == 'train':
        if any(a == '--resume' or a.startswith('--resume=') for a in args.arguments):
            raise SystemExit('Mage-XUT pilot is fresh-start only; do not restore former optimizer state')
        # Keep this entrypoint on argv[0] so Lightning relaunches the same code on rank 1.
        from sakurahdm import train_hdm
        train_hdm.build_model = install()
        from hdm_hybrid_cmuon import configure_hybrid
        stock_optimizer = train_hdm.HDMFlowModule.configure_optimizers

        def configure_optimizers(module):
            if 'hybrid_cmuon' in module.training_config:
                return configure_hybrid(module.model, module.training_config)
            return stock_optimizer(module)

        train_hdm.HDMFlowModule.configure_optimizers = configure_optimizers
        stock_train = train_hdm.train

        def train(config, resume=None):
            if 'mage' not in config['model']['vae'] or resume is not None:
                raise ValueError('Expected Mage stage and a fresh optimizer')
            if os.environ.get('LOCAL_RANK', '0') == '0' and Path(config['training']['output_dir']).exists():
                raise FileExistsError('Use an unused output directory for the Mage pilot')
            sys.argv.insert(1, 'train')
            try:
                return stock_train(config, resume)
            finally:
                sys.argv.pop(1)
        train_hdm.train = train
        runpy.run_path(str(Path(__file__).with_name('hdm-dtk-train.py')))['main']()
    else:
        runpy.run_path(str(Path(__file__).with_name('hdm-dtk-train.py')))['use_vendor_flash']()
        install()
        from sakurahdm.sample import main as sample
        sample()


if __name__ == '__main__':
    main()
