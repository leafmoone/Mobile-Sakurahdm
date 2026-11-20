"""Mobile-O-style alignment, generation SFT, and joint post-training."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from transformers import get_scheduler

from .models import build_model, build_processor


def _merge(base, override):
    return {
        **base,
        **{
            key: _merge(base[key], value)
            if isinstance(base.get(key), dict) and isinstance(value, dict)
            else value
            for key, value in override.items()
        },
    }


def read_config(paths):
    config = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            config = _merge(config, yaml.safe_load(stream))
    return config


def configure_model(model, training):
    model.requires_grad_(False)
    for name in training["trainable"]:
        model.get_submodule(name).requires_grad_(True)

    if lora := training.get("qwen_lora"):
        from peft import LoraConfig

        model.qwen.model.language_model.add_adapter(LoraConfig(**lora))
    if checkpoint := training.get("init_checkpoint"):
        model.load_state_dict(load_file(checkpoint), strict=False)
    return model


def _messages(prompt, image):
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return [{"role": "user", "content": content}]


def _prompts(processor, prompts, images, chat_kwargs):
    images = images if images is not None else [None] * len(prompts)
    return processor.apply_chat_template(
        [_messages(prompt, image) for prompt, image in zip(prompts, images)],
        tokenize=False,
        add_generation_prompt=True,
        **(chat_kwargs or {}),
    )


def encode_prompts(processor, prompts, images=None, *, chat_kwargs=None, tokenizer_kwargs=None):
    text = _prompts(processor, prompts, images, chat_kwargs)
    if images is None:
        tokenizer = processor.tokenizer
        options = dict(tokenizer_kwargs or {})
        limit = options.pop("max_length", None)
        if limit is None:
            return tokenizer(text, padding=True, return_tensors="pt", **options)
        options.pop("truncation", None)
        pad_multiple = options.pop("pad_to_multiple_of", None)
        encoded = tokenizer(text, return_offsets_mapping=True, **options)
        offsets = encoded.pop("offset_mapping")
        if any(len(ids) > limit for ids in encoded["input_ids"]):
            # Locate user content in the rendered template; only its tokens can be removed.
            marker = "SAKURAHDM_PROMPT_BODY"
            frame, = _prompts(processor, [marker], None, chat_kwargs)
            prefix, suffix = frame.split(marker, 1)
            for i, (rendered, spans) in enumerate(zip(text, offsets)):
                excess = len(spans) - limit
                if excess <= 0:
                    continue
                if not rendered.startswith(prefix) or not rendered.endswith(suffix):
                    raise ValueError("Cannot locate the text prompt within the chat template")
                body_end = len(rendered) - len(suffix)
                body_tokens = [j for j, (left, right) in enumerate(spans)
                               if len(prefix) <= left < right <= body_end]
                if len(body_tokens) < excess:
                    raise ValueError("max_length is too small for the chat template")
                remove = set(body_tokens[-excess:])
                for rows in encoded.values():
                    rows[i] = [token for j, token in enumerate(rows[i]) if j not in remove]
        # Never let padding round the context above its configured text cap.
        if pad_multiple and any((len(ids)+pad_multiple-1)//pad_multiple*pad_multiple > limit
                                for ids in encoded["input_ids"]):
            pad_multiple = None
        return tokenizer.pad(encoded, padding=True, pad_to_multiple_of=pad_multiple, return_tensors="pt")
    return processor(text=text, images=images, padding=True, return_tensors="pt")


def encode_answers(processor, questions, images, answers, *, chat_kwargs=None):
    prefixes = _prompts(processor, questions, images, chat_kwargs)
    text = [prefix + answer + processor.tokenizer.eos_token for prefix, answer in zip(prefixes, answers)]
    inputs = processor(
        text=text,
        images=images,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_text_replacement_offsets=True,
    )

    offsets = inputs.pop("offset_mapping")
    replacements = inputs.pop("text_replacement_offsets")
    # Offset mappings are in expanded text coordinates, after visual placeholders.
    boundaries = [
        len(prefix) + sum(
            (entry["new_span"][1] - entry["new_span"][0]) - (entry["span"][1] - entry["span"][0])
            for entry in entries
            if entry["span"][1] <= len(prefix)
        )
        for prefix, entries in zip(prefixes, replacements)
    ]
    ignore = (offsets[..., 1] <= offsets.new_tensor(boundaries)[:, None]) | ~inputs["attention_mask"].bool()
    labels = inputs["input_ids"].masked_fill(ignore, -100)
    return inputs, labels


def image_tensor(images, target_size, device, *, non_blocking=False):
    height, width = target_size
    arrays = [
        np.asarray(ImageOps.fit(image.convert("RGB"), (width, height), Image.Resampling.BICUBIC))
        for image in images
    ]
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=non_blocking).div_(127.5).sub_(1)


def _flow_batch(model, examples, training, device, *, editing):
    target = image_tensor(
        [example["target_image"] for example in examples],
        training["target_size"],
        device,
        non_blocking=training.get("non_blocking", False),
    )
    with torch.no_grad():
        clean = model.encode_images(target)
        if editing:
            source = image_tensor(
                [example["source_image"] for example in examples],
                training["target_size"],
                device,
                non_blocking=training.get("non_blocking", False),
            )
            source = model.encode_images(source)
        else:
            source = None
    t = torch.rand(len(examples), device=device)
    sigma = t.to(clean.dtype).view(-1, *([1] * (clean.ndim - 1)))
    noise = torch.randn_like(clean)
    return torch.lerp(clean, noise, sigma), t, noise - clean, source


def _batch_loss(wrapped, model, processor, examples, training, device):
    task = examples[0]["task"]
    flow = task in {"t2i", "joint", "edit", "joint_edit"}
    language = task in {"understand", "joint", "joint_edit"}
    editing = task in {"edit", "joint_edit"}
    non_blocking = training.get("non_blocking", False)
    chat_kwargs = training.get("chat_kwargs")

    if language:
        language_inputs, labels = encode_answers(
            processor,
            [example["question"] for example in examples],
            [example["source_image"] for example in examples],
            [example["answer"] for example in examples],
            chat_kwargs=chat_kwargs,
        )
        language_inputs = language_inputs.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
    else:
        language_inputs, labels = None, None

    metrics = {}
    if flow:
        prompts = [example["prompt"] for example in examples]
        if dropout := training.get("prompt_dropout"):
            dropped = torch.rand(len(prompts)) < dropout
            prompts = ["" if drop else prompt for prompt, drop in zip(prompts, dropped)]
        inputs = encode_prompts(
            processor,
            prompts,
            [example["source_image"] for example in examples] if editing else None,
            chat_kwargs=chat_kwargs,
            tokenizer_kwargs=training.get("tokenizer_kwargs"),
        ).to(device, non_blocking=non_blocking)
        noisy, t, velocity, source = _flow_batch(model, examples, training, device, editing=editing)
        result = wrapped(
            inputs,
            noisy_latents=noisy,
            timesteps=t,
            source_latents=source,
            language_inputs=language_inputs,
            labels=labels,
        )
        if language:
            prediction, metrics["language"] = result
        else:
            prediction = result
        metrics["flow"] = F.mse_loss(prediction.float(), velocity.float())
    elif language:
        metrics["language"] = wrapped(language_inputs, labels=labels)

    loss = sum(training.get(f"{name}_weight", 1.0) * value for name, value in metrics.items())
    return loss, {"loss": loss.detach(), **{name: value.detach() for name, value in metrics.items()}}


class Progress:
    step = 0

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]


def _checkpoints(accelerator, model):
    def save(models, weights, path):
        if accelerator.is_main_process:
            trainable = {
                name: param.detach().cpu().contiguous()
                for name, param in model.named_parameters() if param.requires_grad
            }
            save_file(trainable, Path(path) / "trainable.safetensors")
        weights.clear()

    def load(models, path):
        model.load_state_dict(load_file(Path(path) / "trainable.safetensors"), strict=False)
        models.clear()

    accelerator.register_save_state_pre_hook(save)
    accelerator.register_load_state_pre_hook(load)


def train(config, resume=None):
    from .data import make_loader

    training = config["training"]
    accelerator = Accelerator(
        mixed_precision=training["mixed_precision"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(**training.get("ddp", {}))],
    )
    set_seed(training["seed"], device_specific=True)

    processor = build_processor(config["model"])
    model = configure_model(build_model(config["model"]), training)
    model.train()
    for submodule in (model.qwen, model.dit, model.vae):
        if not any(param.requires_grad for param in submodule.parameters()):
            submodule.eval()
    if training.get("qwen_gradient_checkpointing"):
        model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if training.get("dit_gradient_checkpointing"):
        model.dit.enable_gradient_checkpointing()

    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, **training["optimizer"])
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_training_steps=training["max_steps"],
        **training["scheduler"],
    )
    wrapped, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    progress = Progress()
    accelerator.register_for_checkpointing(progress)
    _checkpoints(accelerator, model)
    if resume is not None:
        accelerator.load_state(resume)

    # The SDS client already partitions its worker IDs across ranks. Do not
    # re-shard it with Accelerator.prepare; its shard cursor is service-owned.
    loader = make_loader(
        config["data"],
        batch_size=training["batch_size"],
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    output = Path(training["output_dir"])
    last_saved_at = progress.step
    if progress.step < training["max_steps"]:
        for examples in loader:
            with accelerator.accumulate(wrapped):
                loss, metrics = _batch_loss(
                    wrapped, model, processor, examples, training, accelerator.device,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients and (max_norm := training.get("grad_clip")):
                    accelerator.clip_grad_norm_(parameters, max_norm)
                optimizer.step()
                if accelerator.sync_gradients and not optimizer.step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue
            progress.step += 1
            if (interval := training.get("log_steps")) and progress.step % interval == 0:
                reduced = accelerator.reduce(metrics, reduction="mean")
                accelerator.print(
                    f"step={progress.step} " + " ".join(f"{name}={value.item():.5f}" for name, value in reduced.items())
                )
            if (interval := training.get("checkpoint_steps")) and progress.step % interval == 0:
                accelerator.save_state(str(output / f"checkpoint-{progress.step}"))
                last_saved_at = progress.step
            if progress.step >= training["max_steps"]:
                break
    if progress.step > last_saved_at:
        accelerator.save_state(str(output / f"checkpoint-{progress.step}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(read_config([args.config, *args.stage]), args.resume)


if __name__ == "__main__":
    main()
