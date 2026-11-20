"""Reuse SakuraMoon data rules without its SakuraMoon-model tensor contracts."""

import io
import os
import sys
import warnings
from pathlib import Path

from PIL import Image


def sakuramoon_root():
    if configured := os.environ.get("SAKURAMOON_ROOT"):
        root = Path(configured).expanduser().resolve()
        src = str(root / "src")
        # Leave preinstalled vendor kernels ahead of SakuraMoon's shims.
        if src not in sys.path:
            sys.path.append(src)
        return root
    import sakuramoon

    return Path(sakuramoon.__file__).parents[2]


def load_native_data(settings):
    root = sakuramoon_root()
    from sakuramoon.config.load import _Loader, _deep_merge

    # Keep native TOML inheritance without resolving unrelated model/run bindings.
    loader = _Loader(root)
    config = {}
    for reference in settings["config_paths"]:
        config = _deep_merge(config, loader.load(Path(reference)))
    for name in ("spatial_crop", "camera_viewport"):
        policy = config["data"].get(name, {})
        if policy.get("enabled") and policy.get("probability", 0):
            warnings.warn(
                f"SakuraMoon {name} is not used by Mobile-Sakurahdm's HDM geometry.",
                stacklevel=2,
            )
    return config


def leased_samples(path):
    import webdataset as wds

    # SDS owns all rank/worker sharding: do not re-split an issued lease.
    return wds.WebDataset(
        [str(path)], shardshuffle=False, nodesplitter=None, workersplitter=None
    )


class NativeSampleProcessor:
    def __init__(self, config, options, target_size, *, seed, tokenizer_config, max_text_tokens):
        from sakuramoon.config.schema import DataTransparentBackgroundConfig
        from sakuramoon.data.metadata import MetadataFieldMapping
        from transformers import AutoTokenizer

        from .caption import SakuraMoonCaptions, caption_framing

        self.captions = SakuraMoonCaptions(config["caption"])
        self.transparent = DataTransparentBackgroundConfig(
            **config["data"]["transparent_background"]
        )
        self.metadata = MetadataFieldMapping(id_field=options["id_field"])
        self.stage = options["stage"]
        self.seed = seed
        self.short_edge = min(target_size)
        self.no_upscale = config["data"]["image"]["no_upscale"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config["name_or_path"],
            **tokenizer_config.get("tokenizer_kwargs", {}),
        )
        self.framing = caption_framing(self.tokenizer)
        self.max_text_tokens = max_text_tokens

    def prepare(self, metadata, raw, *, key, shard, bucketed):
        from sakuramoon.data.image_ops import ImageMetadataError, normalize_image
        from sakuramoon.data.metadata import parse_shard_metadata
        from sakuramoon.data.pipeline import (
            _MAX_DECODE_DIMENSION,
            _MAX_DECODE_PIXELS,
            _image_bytes,
            _probe_image_dimensions,
            rng_identity,
        )
        from sakuramoon.data.production import adapt_modelscope_metadata
        from sakuramoon.data.transparent_white import apply_transparent_white

        mapped = adapt_modelscope_metadata(metadata)
        sample = parse_shard_metadata(
            mapped, fields=self.metadata, sample_key=f"{shard.record.path}\0{key}"
        )
        rng = rng_identity(
            base_seed=self.seed, stage=self.stage, cycle_index=shard.cycle_index,
            sample_id=sample.id,
        )
        fields = self.captions.fields(metadata)
        payload = _image_bytes(raw)
        try:
            width, height = _probe_image_dimensions(payload)
            if width * height > _MAX_DECODE_PIXELS or max(width, height) > _MAX_DECODE_DIMENSION:
                return None
            with Image.open(io.BytesIO(payload)) as decoded:
                if self.transparent.enabled:
                    result = apply_transparent_white(decoded, fields, self.transparent)
                    if result.outcome.is_reject:
                        return None
                    decoded, fields = result.image, result.fields
                target = normalize_image(decoded)
        except (OSError, SyntaxError, ImageMetadataError, Image.DecompressionBombError):
            return None
        # EXIF-normalized dimensions are used whether or not ARB is enabled.
        if self.no_upscale and not bucketed and min(target.size) < self.short_edge:
            return None

        plan = self.captions.plan(fields, rng.caption_seed)
        serialized = self.captions.serialize(
            plan, self.tokenizer, self.framing, max_text_tokens=self.max_text_tokens
        )
        dense_length = serialized.dense_length
        if self.max_text_tokens is not None:
            dense_length = min(dense_length, self.max_text_tokens)
        return {
            "target_image": target,
            "prompt": serialized.body,
            "serialized_caption": serialized,
            "caption_prepared": True,
            "crop_seed": rng.crop_seed,
            "caption_seed": rng.caption_seed,
            "dense_length": dense_length,
            "sample_id": rng.sample_id,
            "cycle_index": rng.cycle_index,
            "source_shard": shard.record.path,
        }
