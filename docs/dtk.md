# Hygon DCU / DTK

The default is dense PyTorch SDPA and BF16 with your machine's existing Torch/DTK build. Do not replace the vendor wheel with a stock CUDA wheel. VAE encoding stays frozen and in FP32 in the example config. No custom kernel is required.

- Start eager. On an equivalent test input and checkpoint, warm up, then compare median seconds per optimizer step, peak memory and loss against a run with `configs/dtk-compile.yaml`. That optional preset borrows the **setting**, not an expected speedup, from [SakuraMoon's G1 recipe](https://github.com/leafmoone/sakuramoon/blob/08ed12c67a271463396d6402e750ed8746c4bd7d/config/train_g1.toml). Measure after compilation and separately measure start-up time.
- If the DTK Torch build exposes ROCm TunableOp, tune this workload separately. A later run can use `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=0`, and a configured `PYTORCH_TUNABLEOP_FILENAME` generated for the same model shapes, dtype and runtime. Do not reuse SakuraMoon's tuning CSV blindly.
- [SakuraMoon's packed varlen path, combined QK/RoPE/modulation kernels, optimizer and FP8 experiments](https://github.com/leafmoone/sakuramoon) are not default imports: XUT uses U-shaped pooling and a different token/context layout, and Qwen3.5's hybrid backbone is different again. Port independent settings or data pipeline practices before attempting fused kernels.

No Hygon timings, numerical parity or mobile runtime measurements are claimed for this repository.
