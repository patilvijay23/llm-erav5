from pathlib import Path

from p7.data import MemmapTokenPool, PoolManifest, ShardRecord, TokenShardWriter


def test_writer_and_memmap_roundtrip(tmp_path: Path):
    writer = TokenShardWriter(
        tmp_path,
        target_tokens=50,
        block_size=10,
        shard_sequences=2,
    )
    assert writer.append(list(range(17))) == 17
    assert writer.append(list(range(17, 60))) == 33
    shards = writer.finish()
    assert [s.tokens for s in shards] == [20, 20, 10]

    manifest = PoolManifest(
        format_version=2,
        source_name="test",
        dataset_id="test",
        dataset_config="test",
        dataset_revision_requested=None,
        dataset_revision_resolved=None,
        tokenizer_name_or_path="test",
        tokenizer_revision=None,
        tokenizer_class="test",
        tokenizer_vocab_size=100,
        tokenizer_fingerprint="abc",
        block_size=10,
        dtype="uint32",
        target_tokens=50,
        actual_tokens=50,
        total_sequences=5,
        eval_sequences=1,
        train_sequences=4,
        eos_token_id=1,
        seed=1,
        document_sampling="test",
        documents_consumed=1,
        train_tokens=40,
        eval_tokens=10,
        shards=shards,
    )
    manifest.to_json(tmp_path / "manifest.json")
    pool = MemmapTokenPool(tmp_path / "manifest.json")
    assert pool.train_sequences * pool.block_size == 40
    assert pool.eval_sequences * pool.block_size == 10
    assert pool.get_train(0).tolist() == list(range(10))
    assert pool.get_train(3).tolist() == list(range(30, 40))
    assert pool.get_eval(0).tolist() == list(range(40, 50))
