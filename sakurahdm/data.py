"""Read Mobile-O training examples from SakuraDatasetService leases."""

import io
import json
import random
import tarfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from PIL import Image
from sakuradatasetservice import DataServiceClient, RankedDataServiceClient
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _samples(path, wanted):
    # WebDataset TARs keep each sample's members adjacent. The part after
    # the first dot can itself contain dots (for example, gen_image.png).
    with tarfile.open(path, "r|*") as archive:
        key, members = None, {}
        for entry in archive:
            if not entry.isfile():
                continue
            name = PurePosixPath(entry.name)
            stem, delimiter, suffix = name.name.partition(".")
            if not delimiter or not stem:
                continue
            sample_key = str(name.with_name(stem))
            if sample_key != key:
                if members:
                    yield key, members
                key, members = sample_key, {}
            if suffix in wanted:
                with archive.extractfile(entry) as member:
                    members[suffix] = member.read()
        if members:
            yield key, members


def _select(raw, selectors, metadata):
    for member, *keys in selectors:
        value = raw.get(member)
        if value is None:
            continue
        if keys:
            if member not in metadata:
                metadata[member] = json.loads(value)
            value = metadata[member]
            for key in keys:
                if value is None:
                    break
                value = value.get(key)
        if value:
            return value
    return None


def _image(payload, *, native=False):
    if payload is not None:
        with Image.open(io.BytesIO(payload)) as image:
            if native:
                from sakuramoon.data.image_ops import normalize_image

                return normalize_image(image)
            return image.convert("RGB")
    return None


def _text(payload):
    return payload.decode().strip() if isinstance(payload, bytes) else payload


def _question_answer(fields):
    qna = _text(fields.get("qna"))
    if qna:
        pairs = [
            tuple(text.strip() for text in line.removeprefix("Q:").split("A:", 1))
            for line in qna.splitlines()
            if line.startswith("Q:") and "A:" in line
        ]
        if pairs:
            return random.choice(pairs)
    return _text(fields.get("question")), _text(fields.get("answer"))


class LeasedDataset(IterableDataset):
    def __init__(self, config, rank, world_size, *, native=None):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.native = native
        self.fields = {
            name: tuple(
                selector.split("/")
                for selector in ([selectors] if isinstance(selectors, str) else selectors)
            )
            for name, selectors in config.get("fields", {}).items()
        }
        self.members = {
            selector[0] for selectors in self.fields.values() for selector in selectors
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        config = self.config
        delegate = DataServiceClient(
            Path(config["socket_path"]).expanduser().resolve(),
            worker_count=workers * self.world_size,
            request_timeout_seconds=float(config["request_timeout_seconds"]),
        )
        client = RankedDataServiceClient(
            delegate, rank=self.rank, workers_per_rank=workers, world_size=self.world_size
        )
        native_options = config.get("sakuramoon")
        native = None
        if self.native:
            from sakuramoon.data.pipeline import PipelineSampleRejected

            from .sakuramoon_data import NativeSampleProcessor, leased_samples

            native = NativeSampleProcessor(**self.native)

        for lease in client.iter_leases(worker_id):
            with lease as shard:
                rows = (
                    ((raw["__key__"], raw) for raw in leased_samples(shard.local_path))
                    if native else _samples(shard.local_path, self.members)
                )
                for key, raw in rows:
                    metadata = {}
                    fields = {
                        name: _select(raw, selectors, metadata)
                        for name, selectors in self.fields.items()
                        if not native or name not in {"prompt", "target_image"}
                    }
                    question, answer = _question_answer(fields)
                    row = {
                        "task": config["task"],
                        "key": key,
                        "source_image": _image(fields.get("source_image"), native=bool(native)),
                        "question": question,
                        "answer": answer,
                    }
                    if native:
                        metadata_member = native_options["metadata_member"]
                        if metadata_member not in metadata:
                            metadata[metadata_member] = json.loads(raw[metadata_member])
                        try:
                            prepared = native.prepare(
                                metadata[metadata_member], raw, key=key,
                                shard=shard, bucketed=config.get("buckets", {}).get("enabled", False),
                            )
                        except (OSError, PipelineSampleRejected):
                            continue
                        if prepared is None:
                            continue
                        row.update(prepared)
                    else:
                        row["target_image"] = _image(fields.get("target_image"))
                        row["prompt"] = _text(fields.get("prompt"))
                    yield row


def _bucket_shapes(config, target_size, alignment):
    from sakuramoon.data.buckets import generate_base_buckets, scale_buckets

    base = generate_base_buckets(SimpleNamespace(**config))
    return tuple(
        shape for shape in scale_buckets(base, min(target_size))
        if shape.height % alignment == 0 and shape.width % alignment == 0
    )


class ShapeBatches(IterableDataset):
    def __init__(
        self, samples, shapes, *, batch_size, drop_last, window_batches, min_crop_retention,
    ):
        self.samples = samples
        self.shapes = shapes
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.window_size = batch_size * window_batches
        self.min_crop_retention = min_crop_retention

    def _split(self, rows):
        rows.sort(key=lambda row: (
            row.get("dense_length", 0), row.get("cycle_index", 0),
            row.get("caption_seed", 0), row.get("sample_id", 0), row.get("source_shard", ""),
        ))
        return [rows[start:start + self.batch_size] for start in range(0, len(rows), self.batch_size)]

    def __iter__(self):
        if self.shapes is not None:
            from sakuramoon.data.buckets import BucketRejection, assign_bucket

        # Keep each worker's shape/length windows across its disjoint SDS leases.
        pending = {}
        for sample in self.samples:
            size = None
            if self.shapes is not None:
                image = sample["target_image"] or sample["source_image"]
                assignment = assign_bucket(
                    *image.size, self.shapes, min_crop_retention=self.min_crop_retention
                )
                if isinstance(assignment, BucketRejection):
                    continue
                size = assignment.bucket.height, assignment.bucket.width
                sample["bucket_size"] = size
                sample["bucket_assignment"] = assignment
            window = pending.setdefault(size, [])
            window.append(sample)
            if len(window) == self.window_size:
                groups = self._split(pending.pop(size))
                rotation = sum(row.get("caption_seed", 0) for row in window) % len(groups)
                yield from groups[rotation:] + groups[:rotation]
        if not self.drop_last:
            for size in sorted(pending):
                yield from self._split(pending[size])


def _collate(examples):
    return examples


def make_loader(
    config, *, batch_size, rank=0, world_size=1, target_size=None, alignment=None,
    seed=None, tokenizer_config=None,
):
    workers = config.get("num_workers", 0)
    drop_last = config.get("drop_last", False)
    bucket_config = config.get("buckets", {})
    bucketed = bucket_config.get("enabled", False)
    native = None
    window_batches = 1
    loader_kwargs = dict(config.get("loader_kwargs", {}))
    if options := config.get("sakuramoon"):
        from .sakuramoon_data import load_native_data

        native_config = load_native_data(options)
        native_data = native_config["data"]
        native = dict(
            config=native_config, options=options, target_size=target_size, seed=seed,
            tokenizer_config=tokenizer_config,
        )
        window_batches = native_data["loader"]["length_sort_window_batches"]
        drop_last = config.get("drop_last", native_data["loader"]["drop_last"])
        loader_kwargs.setdefault("pin_memory", native_data["loader"]["pin_memory"])
        bucket_config = (
            native_data["buckets"]
            | {"min_crop_retention": native_data["image"]["min_crop_retention"]}
            | bucket_config
        )
    dataset = LeasedDataset(config, rank, world_size, native=native)
    custom_batches = bucketed or native is not None
    if custom_batches:
        dataset = ShapeBatches(
            dataset, _bucket_shapes(bucket_config, target_size, alignment) if bucketed else None,
            batch_size=batch_size, drop_last=drop_last, window_batches=window_batches,
            min_crop_retention=float(bucket_config["min_crop_retention"]) if bucketed else None,
        )
    return DataLoader(
        dataset,
        batch_size=None if custom_batches else batch_size,
        num_workers=workers,
        drop_last=drop_last and not custom_batches,
        collate_fn=_collate,
        **loader_kwargs,
    )
