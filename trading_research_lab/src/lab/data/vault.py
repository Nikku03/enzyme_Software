from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .validators import validate_ohlcv


def compute_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_dataset_id(source_url: str, retrieval_time: str, content_hash: str) -> str:
    payload = f"{source_url}|{retrieval_time}|{content_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class DatasetMetadata:
    dataset_id: str
    source_url: str
    retrieval_time: str
    content_hash: str
    symbol: Optional[str]
    row_count: int
    columns: list[str]
    file_ext: str
    processed_format: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_url": self.source_url,
            "retrieval_time": self.retrieval_time,
            "content_hash": self.content_hash,
            "symbol": self.symbol,
            "row_count": self.row_count,
            "columns": self.columns,
            "file_ext": self.file_ext,
            "processed_format": self.processed_format,
        }


class DataVault:
    def __init__(self, root: Path, processed_format: str = "parquet") -> None:
        self.root = root
        self.processed_format = processed_format
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.meta_dir = self.root / "meta"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        source_url: str,
        content: bytes,
        parser,
        symbol: Optional[str] = None,
        retrieval_time: Optional[str] = None,
        file_ext: str = "csv",
    ) -> DatasetMetadata:
        retrieval_time = retrieval_time or datetime.utcnow().isoformat()
        content_hash = compute_hash_bytes(content)
        dataset_id = compute_dataset_id(source_url, retrieval_time, content_hash)

        raw_path = self.raw_dir / f"{dataset_id}.{file_ext}"
        raw_path.write_bytes(content)

        df = parser(content)
        ok, issues = validate_ohlcv(df)
        if not ok:
            raise ValueError(f"Dataset validation failed: {issues}")

        processed_path = self.processed_dir / f"{dataset_id}.{self.processed_format}"
        if self.processed_format == "parquet":
            df.to_parquet(processed_path)
        else:
            df.to_csv(processed_path)

        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            source_url=source_url,
            retrieval_time=retrieval_time,
            content_hash=content_hash,
            symbol=symbol,
            row_count=len(df),
            columns=list(df.columns),
            file_ext=file_ext,
            processed_format=self.processed_format,
        )
        meta_path = self.meta_dir / f"{dataset_id}.json"
        meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
        return metadata

    def load(self, dataset_id: str) -> pd.DataFrame:
        meta_path = self.meta_dir / f"{dataset_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found for {dataset_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        processed_path = self.processed_dir / f"{dataset_id}.{meta['processed_format']}"
        if meta["processed_format"] == "parquet":
            df = pd.read_parquet(processed_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        return pd.read_csv(processed_path, parse_dates=["date"], index_col="date")

    def get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        meta_path = self.meta_dir / f"{dataset_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found for {dataset_id}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def validate(self, dataset_id: str) -> Dict[str, Any]:
        meta_path = self.meta_dir / f"{dataset_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found for {dataset_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_path = self.raw_dir / f"{dataset_id}.{meta['file_ext']}"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file missing for {dataset_id}")
        content_hash = compute_hash_bytes(raw_path.read_bytes())
        meta_ok = content_hash == meta["content_hash"]
        processed_path = self.processed_dir / f"{dataset_id}.{meta['processed_format']}"
        processed_ok = processed_path.exists()
        return {
            "dataset_id": dataset_id,
            "metadata_hash_matches": meta_ok,
            "processed_exists": processed_ok,
        }

    def list_datasets(self) -> list[str]:
        return [p.stem for p in self.meta_dir.glob("*.json")]
