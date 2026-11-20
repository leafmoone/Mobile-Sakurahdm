"""Lightning training with HDM's logit-normal flow objective and native crop geometry."""
import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import get_scheduler

from .data import make_loader
from .geometry import bucketed_crop, shifted_square_crop
from .models import build_model, build_processor
from .train import configure_model, encode_answers, encode_prompts, image_tensor, read_config


def flow_batch(clean):
    noise = torch.randn_like(clean)
    t = torch.randn(clean.shape[0], device=clean.device).sigmoid()
    sigma = t.reshape(-1, 1, 1, 1)
    return (1-sigma)*clean + sigma*noise, t, noise-clean


def caption_inputs(processor, examples, training, sources=None):
    if all(example.get("caption_prepared") for example in examples) and sources is None:
        return processor.tokenizer.pad([
            {"input_ids": list(e["serialized_caption"].input_ids),
             "attention_mask": list(e["serialized_caption"].attention_mask)} for e in examples
        ], padding=True, return_tensors="pt")
    prompts = [e["prompt"] for e in examples]
    if probability := training.get("prompt_dropout", 0):
        prompts = ["" if torch.rand(()).item() < probability else p for p in prompts]
    return encode_prompts(processor, prompts, sources, chat_kwargs=training.get("chat_kwargs"),
                          tokenizer_kwargs=training.get("tokenizer_kwargs"))


class HDMFlowModule(pl.LightningModule):
    def __init__(self, model, processor, config):
        super().__init__()
        self.model, self.processor, self.config = model, processor, config
        self.training_config = config["training"]

    def train(self, mode=True):
        super().train(mode)
        for part in (self.model.vae, self.model.qwen, self.model.dit):
            if not any(p.requires_grad for p in part.parameters()):
                part.eval()
        return self

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Native frozen caption/bucket records and PIL images stay on the host.
        return batch

    def training_step(self, examples, batch_idx):
        cfg, model = self.training_config, self.model
        task = examples[0]["task"]
        editing = task in {"edit", "joint_edit"}
        language = task in {"understand", "joint", "joint_edit"}
        metrics = {}
        if task != "understand":
            images = [e["target_image"] for e in examples]
            sources = [e["source_image"] for e in examples] if editing else None
            seeds = [e["crop_seed"] for e in examples] if all("crop_seed" in e for e in examples) else None
            crop_options = dict(source_images=sources, crop_seeds=seeds)
            if "bucket_assignment" in examples[0]:
                crops, source_crops, pos, aspect = bucketed_crop(images,
                    [e["bucket_assignment"] for e in examples], model.latent_scale_factor, **crop_options)
            else:
                crops, source_crops, pos, aspect = shifted_square_crop(images,
                    cfg["target_size"], model.latent_scale_factor, **crop_options)
            size = (crops[0].height, crops[0].width)
            clean = model.encode_images(image_tensor(crops, size, self.device))
            source = None if source_crops is None else model.encode_images(image_tensor(source_crops, size, self.device))
            noisy, t, target = flow_batch(clean)
            inputs = caption_inputs(self.processor, examples, cfg, sources).to(self.device)
            predicted = model(inputs, noisy_latents=noisy, timesteps=t, source_latents=source,
                              pos_map=pos, aspect_ratio=aspect)
            metrics["flow"] = F.mse_loss(predicted.float(), target.float())
        if language:
            inputs, labels = encode_answers(self.processor, [e["question"] for e in examples],
                [e["source_image"] for e in examples], [e["answer"] for e in examples],
                chat_kwargs=cfg.get("chat_kwargs"))
            metrics["language"] = model(inputs.to(self.device), labels=labels.to(self.device))
        loss = sum(cfg.get(name+"_weight", 1.0)*value for name, value in metrics.items())
        self.log("train/loss", loss, on_step=True, batch_size=len(examples), sync_dist=True)
        return loss

    def configure_optimizers(self):
        cfg = self.training_config
        connector = list(self.model.connector.parameters())
        ids = {id(p) for p in connector}
        groups = [{"params": [p for p in self.model.parameters() if p.requires_grad and id(p) not in ids]},
                  {"params": [p for p in connector if p.requires_grad],
                   "lr": cfg.get("connector_lr", cfg["optimizer"]["lr"])}]
        optimizer = torch.optim.AdamW([g for g in groups if g["params"]], **cfg["optimizer"])
        schedule = dict(cfg["scheduler"])
        minimum = schedule.pop("min_lr_ratio", None)
        if minimum is not None:
            schedule["name"] = "cosine_with_min_lr"
            schedule["scheduler_specific_kwargs"] = {"min_lr_rate": minimum}
        scheduler = get_scheduler(optimizer=optimizer, num_training_steps=cfg["max_steps"], **schedule)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self):
        cfg = self.training_config
        qwen = self.config["model"]["qwen"]
        return make_loader(self.config["data"], batch_size=cfg["batch_size"],
            rank=self.global_rank, world_size=self.trainer.world_size,
            target_size=cfg["target_size"],
            alignment=self.model.latent_scale_factor*self.model.dit.config.patch_size,
            seed=cfg["seed"], tokenizer_config={"name_or_path": qwen["name_or_path"],
                                               "tokenizer_kwargs": qwen.get("processor_kwargs", {})})

    def on_train_start(self):
        if total := self.training_config.get("global_batch_size"):
            micro = self.training_config["batch_size"] * self.trainer.world_size
            if total % micro:
                raise ValueError("global_batch_size must be divisible by batch_size * world_size")
            self.trainer.accumulate_grad_batches = total // micro


class Artifacts(pl.Callback):
    def __init__(self, training):
        self.config = training
        self.output = Path(training["output_dir"])
        self.last_logged = self.last_saved = 0

    def save(self, trainer, module):
        step = trainer.global_step
        path = self.output/f"checkpoint-{step}"
        if trainer.is_global_zero:
            path.mkdir(parents=True, exist_ok=True)
            save_file({name: p.detach().cpu().contiguous() for name, p in module.model.named_parameters()
                       if p.requires_grad}, path/"trainable.safetensors")
        trainer.strategy.barrier()
        trainer.save_checkpoint(path/"trainer.ckpt")
        self.last_saved = step

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        log_interval = self.config.get("log_steps", 1)
        if log_interval and step > self.last_logged and step % log_interval == 0:
            if trainer.is_global_zero:
                self.output.mkdir(parents=True, exist_ok=True)
                with (self.output/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, "loss": float(trainer.callback_metrics["train/loss"]),
                        "lr": [g["lr"] for g in trainer.optimizers[0].param_groups]})+"\n")
            self.last_logged = step
        interval = self.config.get("checkpoint_steps", 0)
        if interval and step > self.last_saved and step % interval == 0:
            self.save(trainer, module)

    def on_train_end(self, trainer, module):
        if trainer.global_step > self.last_saved:
            self.save(trainer, module)


def train(config, resume=None):
    cfg = config["training"]
    pl.seed_everything(cfg["seed"], workers=True)
    model = configure_model(build_model(config["model"]), cfg)
    if cfg.get("dit_gradient_checkpointing"):
        model.dit.enable_gradient_checkpointing()
    if cfg.get("qwen_gradient_checkpointing"):
        model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    module = HDMFlowModule(model, build_processor(config["model"]), config)
    options = {"max_steps": cfg["max_steps"], "default_root_dir": cfg["output_dir"],
               "accumulate_grad_batches": cfg.get("gradient_accumulation_steps", 1),
               "gradient_clip_val": cfg.get("grad_clip", 0), "logger": False,
               "enable_checkpointing": False, "use_distributed_sampler": False,
               "precision": {"fp16": "16-mixed", "bf16": "bf16-mixed", "no": "32-true"}.get(cfg.get("mixed_precision"), "32-true")}
    options.update(cfg.get("trainer", {}))
    trainer = pl.Trainer(callbacks=[Artifacts(cfg)], **options)
    trainer.fit(module, ckpt_path=resume)
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--resume", help="Native Lightning trainer.ckpt (optimizer and scheduler resume)")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = read_config([args.config, *args.stage])
    if args.print_config:
        import yaml
        print(yaml.safe_dump(config, sort_keys=False))
        return
    train(config, args.resume)


if __name__ == "__main__":
    main()
