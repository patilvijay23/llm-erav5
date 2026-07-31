"""Data preparation and memory-mapped token pools for P7."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class ShardRecord:
    file: str
    tokens: int
    sequences: int
    sha256: str


@dataclass
class PoolManifest:
    format_version: int
    source_name: str
    dataset_id: str
    dataset_config: str
    dataset_revision_requested: str | None
    dataset_revision_resolved: str | None
    tokenizer_name_or_path: str
    tokenizer_revision: str | None
    tokenizer_class: str
    tokenizer_vocab_size: int
    tokenizer_fingerprint: str
    block_size: int
    dtype: str
    target_tokens: int
    actual_tokens: int
    total_sequences: int
    eval_sequences: int
    train_sequences: int
    eos_token_id: int
    seed: int
    document_sampling: str
    documents_consumed: int
    train_tokens: int = 0
    eval_tokens: int = 0
    per_split_documents: dict[str, int] = field(default_factory=dict)
    per_split_tokens: dict[str, int] = field(default_factory=dict)
    shards: list[ShardRecord] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "PoolManifest":
        raw = json.loads(Path(path).read_text())
        raw["shards"] = [ShardRecord(**item) for item in raw["shards"]]
        raw.setdefault("train_tokens", int(raw["train_sequences"]) * int(raw["block_size"]))
        raw.setdefault("eval_tokens", int(raw["eval_sequences"]) * int(raw["block_size"]))
        return cls(**raw)


def resolve_dataset_revision(dataset_id: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset_id, revision=revision).sha
    except Exception:
        # Preparation can still run in an offline/mirrored environment. The
        # manifest keeps the requested revision and records that resolution
        # was unavailable.
        return None


def tokenizer_fingerprint(tokenizer: "PreTrainedTokenizerBase") -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "vocab": sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]),
        "special_tokens": tokenizer.special_tokens_map,
        "added_vocab": sorted(tokenizer.get_added_vocab().items(), key=lambda item: item[1]),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_tokenizer(
    name_or_path: str,
    *,
    revision: str | None = None,
    expected_vocab_size: int = 196_608,
    allow_vocab_mismatch: bool = False,
) -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        revision=revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id for document packing.")
    if len(tokenizer) != expected_vocab_size and not allow_vocab_mismatch:
        raise ValueError(
            f"Tokenizer vocabulary has {len(tokenizer):,} entries; expected "
            f"{expected_vocab_size:,}. Pass --allow-vocab-mismatch only for a "
            "smoke test, not the reported P7 run."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TokenShardWriter:
    """Stream uint32 token IDs into sequence-aligned shard files."""

    def __init__(
        self,
        output_dir: Path,
        *,
        target_tokens: int,
        block_size: int,
        shard_sequences: int,
    ) -> None:
        if target_tokens % block_size:
            raise ValueError("target_tokens must be divisible by block_size")
        if shard_sequences <= 0:
            raise ValueError("shard_sequences must be positive")
        self.output_dir = output_dir
        self.target_tokens = target_tokens
        self.block_size = block_size
        self.shard_token_capacity = shard_sequences * block_size
        self.tokens_written = 0
        self.shards: list[ShardRecord] = []
        self._handle: Any | None = None
        self._hasher: hashlib._Hash | None = None
        self._current_path: Path | None = None
        self._current_tokens = 0

    @property
    def remaining(self) -> int:
        return self.target_tokens - self.tokens_written

    def _open_shard(self) -> None:
        index = len(self.shards)
        self._current_path = self.output_dir / f"tokens-{index:05d}.bin"
        self._handle = self._current_path.open("wb")
        self._hasher = hashlib.sha256()
        self._current_tokens = 0

    def _close_shard(self) -> None:
        if self._handle is None or self._current_path is None or self._hasher is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        if self._current_tokens % self.block_size:
            raise RuntimeError("Internal error: shard is not sequence aligned")
        self.shards.append(
            ShardRecord(
                file=self._current_path.name,
                tokens=self._current_tokens,
                sequences=self._current_tokens // self.block_size,
                sha256=self._hasher.hexdigest(),
            )
        )
        self._handle = None
        self._hasher = None
        self._current_path = None
        self._current_tokens = 0

    def append(self, token_ids: Sequence[int]) -> int:
        if self.remaining <= 0 or not token_ids:
            return 0
        ids = np.asarray(token_ids, dtype=np.uint32)
        accepted_total = 0
        cursor = 0
        while cursor < len(ids) and self.remaining > 0:
            if self._handle is None:
                self._open_shard()
            assert self._handle is not None and self._hasher is not None
            capacity = min(
                self.shard_token_capacity - self._current_tokens,
                self.remaining,
            )
            take = min(capacity, len(ids) - cursor)
            chunk = np.ascontiguousarray(ids[cursor : cursor + take])
            raw = chunk.tobytes(order="C")
            self._handle.write(raw)
            self._hasher.update(raw)
            self._current_tokens += take
            self.tokens_written += take
            accepted_total += take
            cursor += take
            if self._current_tokens == self.shard_token_capacity:
                # A capacity selected as whole sequences guarantees alignment.
                self._close_shard()
        return accepted_total

    def finish(self) -> list[ShardRecord]:
        if self.tokens_written != self.target_tokens:
            raise RuntimeError(
                f"Prepared {self.tokens_written:,} tokens, expected "
                f"{self.target_tokens:,}. The source stream ended too early."
            )
        self._close_shard()
        if sum(s.tokens for s in self.shards) != self.target_tokens:
            raise RuntimeError("Shard token counts do not sum to target")
        return self.shards


def _batched_documents(
    documents: Iterable[tuple[str, str]], batch_size: int
) -> Iterator[tuple[list[str], list[str]]]:
    texts: list[str] = []
    labels: list[str] = []
    for text, label in documents:
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(text)
        labels.append(label)
        if len(texts) >= batch_size:
            yield texts, labels
            texts, labels = [], []
    if texts:
        yield texts, labels


def fineweb_documents(
    *,
    config: str,
    revision: str | None,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=config,
        split="train",
        streaming=True,
        revision=revision,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for row in dataset:
        yield row["text"], "en"


def indiccorp_documents(
    *,
    revision: str | None,
    seed: int,
    shuffle_buffer: int,
    selected_splits: Sequence[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Uniform-by-document sampling across active IndicCorpV2 language splits.

    This deliberately fixes the internal Indic language policy so that the P7
    variable is the *dataset-level* mixture shift. Splits that exhaust are
    removed. The resulting per-split document/token totals are written to the
    manifest and must be reused by the warm-transition control arm.
    """
    from datasets import get_dataset_split_names, load_dataset

    splits = list(
        selected_splits
        or get_dataset_split_names(
            "ai4bharat/IndicCorpV2",
            config_name="indiccorp_v2",
            revision=revision,
        )
    )
    if not splits:
        raise RuntimeError("IndicCorpV2 exposed no splits")

    iterators: dict[str, Iterator[dict[str, Any]]] = {}
    per_split_shuffle_buffer = max(1_000, shuffle_buffer // max(len(splits), 1))
    for index, split in enumerate(splits):
        stream = load_dataset(
            "ai4bharat/IndicCorpV2",
            name="indiccorp_v2",
            split=split,
            streaming=True,
            revision=revision,
        )
        stream = stream.shuffle(
            seed=seed + index + 1,
            buffer_size=per_split_shuffle_buffer,
        )
        iterators[split] = iter(stream)

    rng = random.Random(seed ^ 0x1D1C)
    active = list(iterators)
    while active:
        split = active[rng.randrange(len(active))]
        try:
            row = next(iterators[split])
        except StopIteration:
            active.remove(split)
            continue
        yield row["text"], split


def prepare_pool(
    *,
    source: str,
    output_dir: str | Path,
    tokenizer_name_or_path: str,
    tokenizer_revision: str | None,
    expected_vocab_size: int,
    allow_vocab_mismatch: bool,
    train_tokens: int,
    block_size: int,
    eval_sequences: int,
    shard_sequences: int,
    tokenize_batch_size: int,
    seed: int,
    shuffle_buffer: int,
    overwrite: bool,
    fineweb_config: str = "sample-100BT",
    fineweb_revision: str | None = None,
    indic_revision: str | None = "984b75b20ce408f9ba27c6558e9279e8e1b6edfd",
    indic_splits: Sequence[str] | None = None,
) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output} is not empty. Use --overwrite to recreate the pool."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    if train_tokens <= 0 or train_tokens % block_size:
        raise ValueError("train_tokens must be positive and divisible by block_size")
    if eval_sequences < 0:
        raise ValueError("eval_sequences must be non-negative")
    train_sequences = train_tokens // block_size
    eval_tokens = eval_sequences * block_size
    target_tokens = train_tokens + eval_tokens
    total_sequences = train_sequences + eval_sequences

    tokenizer = load_tokenizer(
        tokenizer_name_or_path,
        revision=tokenizer_revision,
        expected_vocab_size=expected_vocab_size,
        allow_vocab_mismatch=allow_vocab_mismatch,
    )
    eos = int(tokenizer.eos_token_id)

    if source == "fineweb_edu":
        dataset_id = "HuggingFaceFW/fineweb-edu"
        dataset_config = fineweb_config
        requested_revision = fineweb_revision
        documents = fineweb_documents(
            config=fineweb_config,
            revision=fineweb_revision,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
        )
        sampling = "FineWeb streaming shuffle; stop at exact target-token count"
    elif source == "indiccorp_v2":
        dataset_id = "ai4bharat/IndicCorpV2"
        dataset_config = "indiccorp_v2"
        requested_revision = indic_revision
        documents = indiccorp_documents(
            revision=indic_revision,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
            selected_splits=indic_splits,
        )
        sampling = "uniform-by-document across active language splits"
    else:
        raise ValueError("source must be fineweb_edu or indiccorp_v2")

    writer = TokenShardWriter(
        output,
        target_tokens=target_tokens,
        block_size=block_size,
        shard_sequences=shard_sequences,
    )
    per_split_documents: dict[str, int] = {}
    per_split_tokens: dict[str, int] = {}
    documents_consumed = 0

    for texts, labels in _batched_documents(documents, tokenize_batch_size):
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )["input_ids"]
        for ids, label in zip(encoded, labels, strict=True):
            if writer.remaining <= 0:
                break
            # EOS keeps document boundaries visible after packing.
            candidate = list(ids)
            candidate.append(eos)
            accepted = writer.append(candidate)
            if accepted <= 0:
                break
            documents_consumed += 1
            per_split_documents[label] = per_split_documents.get(label, 0) + 1
            per_split_tokens[label] = per_split_tokens.get(label, 0) + accepted
        if writer.remaining <= 0:
            break

    shards = writer.finish()
    manifest = PoolManifest(
        format_version=2,
        source_name=source,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_revision_requested=requested_revision,
        dataset_revision_resolved=resolve_dataset_revision(
            dataset_id, requested_revision
        ),
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_revision=tokenizer_revision,
        tokenizer_class=tokenizer.__class__.__name__,
        tokenizer_vocab_size=len(tokenizer),
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        block_size=block_size,
        dtype="uint32",
        target_tokens=target_tokens,
        actual_tokens=target_tokens,
        total_sequences=total_sequences,
        eval_sequences=eval_sequences,
        train_sequences=train_sequences,
        eos_token_id=eos,
        seed=seed,
        document_sampling=sampling,
        documents_consumed=documents_consumed,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        per_split_documents=per_split_documents,
        per_split_tokens=per_split_tokens,
        shards=shards,
    )
    manifest_path = output / "manifest.json"
    manifest.to_json(manifest_path)
    return manifest_path


class MemmapTokenPool:
    """Random-access packed sequences backed by uint32 memory maps."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = PoolManifest.from_json(self.manifest_path)
        self.block_size = self.manifest.block_size
        self.train_sequences = self.manifest.train_sequences
        self.eval_sequences = self.manifest.eval_sequences
        self.total_sequences = self.manifest.total_sequences
        self._cumulative: list[int] = []
        running = 0
        for shard in self.manifest.shards:
            running += shard.sequences
            self._cumulative.append(running)
        if running != self.total_sequences:
            raise ValueError("Manifest shard sequence counts are inconsistent")
        self._maps: dict[int, np.memmap] = {}

    def _map(self, shard_index: int) -> np.memmap:
        if shard_index not in self._maps:
            record = self.manifest.shards[shard_index]
            self._maps[shard_index] = np.memmap(
                self.root / record.file,
                mode="r",
                dtype=np.uint32,
                shape=(record.tokens,),
            )
        return self._maps[shard_index]

    def _get_absolute(self, sequence_index: int) -> np.ndarray:
        if not 0 <= sequence_index < self.total_sequences:
            raise IndexError(sequence_index)
        shard_index = bisect_right(self._cumulative, sequence_index)
        prior = 0 if shard_index == 0 else self._cumulative[shard_index - 1]
        local_sequence = sequence_index - prior
        start = local_sequence * self.block_size
        stop = start + self.block_size
        return np.asarray(self._map(shard_index)[start:stop], dtype=np.int64)

    def get_train(self, sequence_index: int) -> torch.Tensor:
        if not 0 <= sequence_index < self.train_sequences:
            raise IndexError(sequence_index)
        return torch.from_numpy(self._get_absolute(sequence_index).copy())

    def get_eval(self, eval_index: int) -> torch.Tensor:
        if not 0 <= eval_index < self.eval_sequences:
            raise IndexError(eval_index)
        absolute = self.train_sequences + eval_index
        return torch.from_numpy(self._get_absolute(absolute).copy())


def validate_compatible_pools(
    fineweb: MemmapTokenPool,
    indic: MemmapTokenPool,
    *,
    required_block_size: int,
) -> None:
    fields = (
        "block_size",
        "tokenizer_vocab_size",
        "tokenizer_fingerprint",
        "eos_token_id",
    )
    for field_name in fields:
        fw_value = getattr(fineweb.manifest, field_name)
        indic_value = getattr(indic.manifest, field_name)
        if fw_value != indic_value:
            raise ValueError(
                f"Pool mismatch for {field_name}: FineWeb={fw_value!r}, "
                f"IndicCorp={indic_value!r}"
            )
    if fineweb.block_size != required_block_size:
        raise ValueError(
            f"Pool block size {fineweb.block_size} != requested "
            f"{required_block_size}"
        )
