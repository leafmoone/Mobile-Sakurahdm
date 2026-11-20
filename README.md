# Mobile-Sakurahdm

实验性训练工程：从 [Mobile-O](https://github.com/Amshaker/Mobile-O) 出发，将生成器换成 [HDM XUT](https://github.com/KohakuBlueleaf/HDM)，将理解与生成共享的多模态语言骨干换成 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，用 [SakuraDatasetService](https://github.com/leafmoone/Sakuradatasetservice) 供数据。

保留 Mobile-O 的层融合、序列细化和宽度投影，不再做 256 槽重采样。XUT 读取全部有效文本 token；同一微批次里按实际长度分组，padding 不进入连接器细化层或 XUT。新连接器从零开始，DiT 默认从 HDM 初始化，不复用 Mobile-O 的 SANA/FastVLM 权重。生成使用 HDM 配套的 EQ-SDXL VAE、逐通道 latent 归一化、flow matching 和 Euler 采样。理解训练只给答案 token 计损失。

## 安装

Linux 上先装与机器匹配的 Torch / TorchVision（DCU 使用现有 DTK wheel），然后在本仓库根目录：

```bash
python -m pip install -r requirements-sakurahdm.txt
git clone https://github.com/KohakuBlueleaf/HDM ../HDM
git -C ../HDM checkout 5fef7c4b71fe8386b497176021fe458810fdb7c0
python -m pip install --no-deps -e ../HDM
python -m pip install omegaconf toml
git clone https://github.com/leafmoone/Sakuradatasetservice ../Sakuradatasetservice
git -C ../Sakuradatasetservice checkout a44499e5326b6f67c0e6ff75f51ad40f715f44e9
python -m pip install -e ../Sakuradatasetservice
```

模型 ID、本地权重路径、dtype、分辨率、数据字段及训练参数都由 YAML 提供；可添加 `--stage` 覆盖相应项，不用改代码。为可复现训练可在每个模型的 `load_kwargs`（以及 Qwen 的 `processor_kwargs`）传入需要的 `revision`。

### XUT 门控激活

训练和采样默认将 XUT 的全部 SwiGLU MLP 换成 [PowLU 的双投影版本](https://arxiv.org/html/2605.25704v1#S3.SS1)（默认 `m=3`，可用 `model.hdm.powlu_m` 调整），不改 Qwen、连接器或 AdaLN 的 Mish。原有 `w12`、`w3` 投影及权重名不变，可加载原 HDM 权重；如需复现 SwiGLU 基线，把 `model.hdm.powlu_m` 设为 YAML 的 `null`。从旧权重继续训练时，权重兼容不意味着切换激活后输出等同。

## HDM 训练

三套脚本都走 HDM 风格的 Lightning flow matching：以 logit-normal 采样时间，目标为 `noise - latent`。TREAD 的 token 路由默认关闭，原有前后段保留，仍可加载原 HDM 权重。对齐阶段冻结 Qwen 与 VAE，训练 XUT 和 Mobile-O 条件连接器（MCP）。

| 脚本 | 分辨率 | 全局 batch | 精度 | XUT / MCP 学习率 | ARB |
|---|---:|---:|---|---|---|
| `scripts/train_512.sh` | 512 | 512 | FP16 mixed | `2e-5 / 2e-4` | 关 |
| `scripts/train_768.sh` | 768 | 384 | FP16 mixed | `2e-5 / 2e-4` | 关 |
| `scripts/train_1024.sh` | 1024 | 256 | FP16 mixed | `1e-5 / 1e-4` | 开 |

表中的学习率是预热后的绝对值，不使用 HDM 示例里的 μP 倍率。三套配置沿用 1000 步预热、余弦降到 1% 的比例、AdamW `betas=(0.9, 0.995)` 和梯度裁剪 1。运行前按显存调节微批次和设备数；梯度累积按目标全局 batch 计算。

`configs/align.yaml` 的原 512 数值不变。原 Accelerate 入口保留用于已有的对齐、生成 SFT、联合 VQA 和配对编辑配置；旧的 `trainable.safetensors` 可用于初始化，旧 Accelerate 优化器状态不作为 Lightning 断点使用。

### SakuraMoon 数据

三套 HDM 配置都使用 SDS 的分片租约和 SakuraMoon 的数据处理，不单独重做 caption 规则。它们从授权的 SakuraMoon checkout 加载 `config/base.toml`（可叠加自己的数据配置），复用 WebDataset 样本读取、元数据解析、图像处理，以及 tags/NL/candidate、画师或角色条件、dropout 和原生 Qwen 序列化。原生数据路径不会再应用一轮 Mobile-O 文本 dropout。文生图训练和采样的文本上限由 `training.tokenizer_kwargs.max_length` 控制，默认 512 token（包括 Qwen 模板和角色/画师条件）；原生数据仍按 SakuraMoon 的完整字段边界处理超长 caption，不把每条文本补到固定长度。启用 ARB 时使用 SakuraMoon 的桶和裁剪，关闭时改为 HDM 随机偏移方形裁剪，caption 不受 ARB 开关影响。SakuraMoon 专用的模型条件向量、训练数据服务内部实现及实验性相机裁剪不属于这个适配层。

运行前启动 SDS，让 `data.socket_path` 指向对应 socket。准备好 SakuraMoon checkout（本文适配时使用 `08ed12c67a271463396d6402e750ed8746c4bd7d`），再运行：

```bash
export SAKURAMOON_ROOT=/path/to/SakuraMoon
bash scripts/train_512.sh --stage /path/to/local-512.yaml
bash scripts/train_768.sh --stage /path/to/local-768.yaml
bash scripts/train_1024.sh --stage /path/to/local-1024.yaml
```

本机配置可改 `data.socket_path`、`training.batch_size`（单卡微批次）、`training.trainer.devices` 和输出路径。后续阶段可通过 `training.init_checkpoint` 加载上一阶段的 `trainable.safetensors`；同阶段用 `--resume /path/to/checkpoint.ckpt` 恢复 Lightning 状态。要关闭 1024 的分桶，在命令末尾加 `--stage configs/no-buckets.yaml`。SDS 维护分片租约，不能把 Lightning 恢复理解成样本级精确重放。Lightning 全量断点比训练权重导出大得多，按磁盘空间调整 `training.checkpoint_steps`。带旧重采样层的 Lightning 断点不能直接 `--resume`，可从旧的 `trainable.safetensors` 初始化同名权重；旧连接器输出条件与新版不同。

## 原版任务配置

| 阶段 | 数据 | 更新参数 |
|---|---|---|
| 对齐 | 大规模 caption + 图；角色和画风平衡集可用作其中一部分 | 连接器、HDM；Qwen、VAE 冻结 |
| 生成 SFT | 另配精选 prompt→图像数据，以构图、属性、关系和指令覆盖为主；不等同于继续喂同一批画师图 | 从对齐接续，仍训连接器、HDM |
| 统一后训练 | 同一样本包含生成 prompt、图、问题、答案 | HDM、连接器、Qwen 文本 LoRA、视觉编码器；VAE 冻结 |

```bash
accelerate launch -m sakurahdm.train configs/align.yaml
accelerate launch -m sakurahdm.train configs/align.yaml --stage configs/generation-sft.yaml
accelerate launch -m sakurahdm.train configs/align.yaml --stage configs/joint.yaml
```

这些 YAML 是实验起点，不是正式训练时长或显存保证。阶段之间用 `training.init_checkpoint` 加载上一阶段的 `trainable.safetensors`；同一阶段继续用 `--resume <checkpoint-dir>` 恢复优化器、调度器、迭代与随机状态。SDS 独立维护 TAR 租约和分片进度；预取和未完成租约意味着不是样本级精确恢复。
联合阶段若要让完整视觉骨干以 FP32 持有可训练参数，可覆盖 `model.qwen.load_kwargs.dtype: float32`，保留训练混精设置；这也会增加文本基座占用。

### 数据服务

按 SDS 文档为各阶段开同机 Unix socket，`data.socket_path` 指向对应服务。服务的 `--workers` = 训练进程数 × `max(1, data.num_workers)`；多卡不额外二次分片。步数制训练推荐 `--repeat`，确保每个 worker 有足够的可用 TAR。角色/画师按父目录分组时，服务端可用 `--sampling balance --per-folder <N>`。

WebDataset TAR 的同样本成员连续存放。字段映射支持成员后缀、JSON 子字段或顺序候选，例如 `jpg`、`json/captions/nl2`、`[avif, jpg]`。理解图映射到 `source_image`；普通联合四元组的理解图可与生成目标共用同一成员。也可配置 `qna: qna.txt` 读取 Mobile-O 的问题/答案。编辑是独立可选任务，用训练好的源图 latent 残差与 Qwen 图像条件；须用真实源图、编辑指令、目标图成对训练，不能从普通 caption 或 HDM 的初始权重直接获得编辑能力。

## 采样

HDM 三套配置采样时沿用训练时的 `SAKURAMOON_ROOT`，先传共享 YAML，再传对应分辨率。若训练期间修改了模型配置（例如 PowLU 参数），采样也要加同一份覆盖配置：

```bash
python -m sakurahdm.sample configs/hdm-base.yaml configs/hdm-1024.yaml \
  --checkpoint /path/to/trainable.safetensors \
  --prompt 'an illustration of a red fox in a snowy forest' --output fox.png
```

原任务配置采样示例：

```bash
python -m sakurahdm.sample configs/align.yaml \
  --checkpoint runs/align/checkpoint-1000/trainable.safetensors \
  --prompt 'an illustration of a red fox in a snowy forest' --output fox.png
```

SFT、联合或编辑采样把对应阶段 YAML 紧跟基础 YAML，传入该阶段的权重；编辑还需 `--source-image`。复现部署前应使用自己的设备和权重验证效果，不沿用 Mobile-O 的 iOS 延迟、内存或质量指标。原版脚本保留作参考，不是此改版的训练入口。

同机加速见 [Hygon / DTK](docs/dtk.md)。

## 来源与许可证

[Mobile-O 原版文档、原作者及引用](UPSTREAM.md)，上游版本 `91c255080a0130846fe5b5cd54ff5af3c05ab2b9`。本仓库对其连接器、训练数据适配及模型组合做了上述改动；保留 [CC BY-NC-SA 4.0](LICENSE.txt)，仅用于非商业用途。HDM 代码与权重按其仓库说明同为 CC BY-NC-SA 4.0；Qwen3.5-2B 和 EQ-SDXL-VAE 按各自模型卡为 Apache-2.0。SDS 仅作为独立安装的依赖，未复制其源码或 SakuraMoon 源码；不在此仓库分发数据或权重。

## 首版验证

云端 CPU 验证覆盖原生微型 Qwen3.5、XUT 与 VAE 的训练前后向、联合 Flow/VQA 与 LoRA、恢复后的权重一致性、采样和独立 SDS 分片测试。另外实读 HDM 官方 340M Diffusers 权重，并完成一次有限输出的 BF16 前向。未跑真实大规模训练、未连接生产数据，也未在目标 DCU 验证速度。SDS 测试在进程内通过真实 TAR、租约处理与 ACK；此云端沙箱无法绑定 Unix socket。
