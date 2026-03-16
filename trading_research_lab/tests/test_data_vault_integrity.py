from pathlib import Path

from lab.data.vault import DataVault, compute_dataset_id, compute_hash_bytes
from lab.research.extractors import parse_stooq_csv


def test_data_vault_integrity(tmp_path: Path):
    content = (
        "Date,Open,High,Low,Close,Volume\n"
        "2020-01-01,10,11,9,10,1000\n"
        "2020-01-02,10,12,9,11,1100\n"
    ).encode("utf-8")
    url = "https://stooq.com/q/d/l/?s=spy.us&i=d"
    retrieval_time = "2020-01-01T00:00:00"

    vault = DataVault(root=tmp_path, processed_format="parquet")
    metadata = vault.ingest(
        source_url=url,
        content=content,
        parser=parse_stooq_csv,
        symbol="SPY",
        file_ext="csv",
        retrieval_time=retrieval_time,
    )

    expected_hash = compute_hash_bytes(content)
    expected_id = compute_dataset_id(url, retrieval_time, expected_hash)
    assert metadata.dataset_id == expected_id

    meta = vault.get_metadata(metadata.dataset_id)
    assert meta["content_hash"] == expected_hash

    validation = vault.validate(metadata.dataset_id)
    assert validation["metadata_hash_matches"] is True
    assert validation["processed_exists"] is True
