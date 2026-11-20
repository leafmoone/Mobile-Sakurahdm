"""Read Mobile-O training examples from SakuraDatasetService leases."""

import io
import json
import random
import tarfile
from pathlib import Path, PurePosixPath

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


def _image(payload):
    if payload is not None:
        with Image.open(io.BytesIO(payload)) as image:
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
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.fields = {
            name: tuple(
                selector.split("/")
                for selector in ([selectors] if isinstance(selectors, str) else selectors)
            )
            for name, selectors in config["fields"].items()
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
        for lease in client.iter_leases(worker_id):
            with lease as shard:
                for key, raw in _samples(shard.local_path, self.members):
                    metadata = {}
                    fields = {
                        name: _select(raw, selectors, metadata)
                        for name, selectors in self.fields.items()
                    }
                    question, answer = _question_answer(fields)
                    yield {
                        "task": config["task"],
                        "key": key,
                        "prompt": _text(fields.get("prompt")),
                        "source_image": _image(fields.get("source_image")),
                        "target_image": _image(fields.get("target_image")),
                        "question": question,
                        "answer": answer,
                    }


def _collate(examples):
    return examples


def make_loader(config, *, batch_size, rank=0, world_size=1):
    workers = config.get("num_workers", 0)
    return DataLoader(
        LeasedDataset(config, rank, world_size),
        batch_size=batch_size,
        num_workers=workers,
        drop_last=config.get("drop_last", False),
        collate_fn=_collate,
        **config.get("loader_kwargs", {}),
    )
