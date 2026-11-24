# Mage Flow VAE and widened XUT pilot

This branch adds an opt-in DTK training entrypoint. The stock model and launchers are unchanged. The experiment uses a frozen BF16 Mage encoder's posterior mean in native latent units (no EQ-SDXL statistics), 128 latent channels with 16× VAE downsampling, and XUT patch size 1. At 512×512, that is still 1,024 image tokens before TREAD. The flow velocity objective and Euler sampler remain those of HDM.

The initializer widens a local HDM-340M checkpoint from 1024 to 1280 hidden channels, 16 to 20 attention heads at head dimension 64, and 3072 to 4096 MLP channels. It copies Q/K/V, gate/value and AdaLN components separately, preserves original RoPE heads and timestep frequencies, and initializes fresh latent I/O layers. It is not a numerically equivalent conversion of the pretrained model. Old 4-channel checkpoints or optimizer states cannot resume this experiment.

## Training and sampling

Set up the DTK environment, the vendor FlashAttention wheel, and local HDM/Qwen paths as in the main project. Set `SAKURAMOON_ROOT` to your local SakuraMoon `08ed12c67a271463396d6402e750ed8746c4bd7d` checkout. Place the [Mage-Flow-Base VAE](https://modelscope.cn/models/microsoft/Mage-Flow-Base) checkpoint in `models/Mage-Flow-Base/vae/` or override its path in your final YAML stage. The Mage stage pins the SHA-256 for ModelScope revision `4b7f1a41b3f7d4eacb8cf7450a23d328bf95fc09`. Do not reuse any EQ-SDXL latent cache.

```bash
python scripts/hdm-mage-xut-run.py train configs/hdm-base.yaml \
  --stage configs/hdm-512.yaml --stage local-models-and-data.yaml \
  --stage configs/dtk-compile.yaml --stage configs/hdm-mage-xut-1280.yaml \
  --stage configs/hdm-mage-cmuon-520.yaml --print-config
```

Use `--print-config` to check paths and settings. For training, use a fresh output directory and a working SDS socket; set a short `training.max_steps` in a final overlay before removing `--print-config`. The pilot launcher deliberately refuses `--resume`. Its `sample` subcommand uses the stock sampling options with the Mage overlay and weights from this architecture. An editing experiment must also enable `model.source_adapter` in its config.

The optional hybrid stage sets two DCUs, per-device microbatch 10, accumulation 26 and global batch 520. XUT attention, MLP and shared AdaLN modulation matrices use semantic-chunk CMuon with BF16 momentum, NS4, Nesterov and Moonlight scaling. Other trainable parameters use dense AdamW, not SakuraMoon's AdamW8bit. The LR multipliers are 1.0 for CMuon and 0.6 for AdamW on the existing HDM XUT and connector rates. The AdamW-only path remains available by omitting this stage and configuring your batch separately.

The packed XUT prototype is separate from this training entrypoint and is not enabled by default. `scripts/check_xut_packed_pilot.py --help` describes the numerical check. The grouped-projection variant passed selected FP16 comparisons without XUT compile; the compile combination did not pass. Do not treat it as a validated production path.

To publish metrics independently of training, append `configs/hdm-swanlab.yaml` and, if wanted, configure `training.preview: {every_steps: 1000, count: 6}`. Install `swanlab==0.10.6` and Pillow in a separate environment. Run `scripts/swanlab_hdm.py RUN_DIR --mode offline` for local-only logging or use online mode with `SWANLAB_API_KEY` and the HTTPS proxy in the environment. Checkpoints are not uploaded. Use `--scalars-only` to omit grids.

## Short throughput check

On 2×64 GiB Hygon BW, 512×512, 50% TREAD, using cached real samples, one warmup update and two timed updates at global batch 520:

| Optimizer | Images/s | Seconds/update | Peak allocated / reserved per device |
|---|---:|---:|---:|
| AdamW | 8.92 | 58.32 | 49.67 / 62.60 GiB |
| CMuon + dense AdamW | 9.01 | 57.70 | 46.70 / 62.56 GiB |

The differences do not establish a general speedup; reserved memory still approached card capacity. The test did not include live data downloads, saving or sampling. The VAE and width conversion were smoke-tested, not validated for long-run generation quality. Run the local unit tests with `python -m unittest discover -s tests -v` (PyTorch and Pillow required).
