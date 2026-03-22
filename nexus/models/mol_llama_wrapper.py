from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer

    _TRANSFORMERS_OK = True
except Exception:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None
    _TRANSFORMERS_OK = False


@dataclass
class LatentBlueprint:
    sequence: torch.Tensor
    pooled: torch.Tensor
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    smiles: str
    source: str
    chirality_signature: torch.Tensor

    def to(self, device: torch.device | str) -> "LatentBlueprint":
        return LatentBlueprint(
            sequence=self.sequence.to(device),
            pooled=self.pooled.to(device),
            token_ids=self.token_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            smiles=self.smiles,
            source=self.source,
            chirality_signature=self.chirality_signature.to(device),
        )


class _FallbackTokenizer:
    def __init__(self) -> None:
        vocab = [
            "<pad>",
            "<bos>",
            "<eos>",
        ] + sorted(set("#%()+-./0123456789:=@ABCFHILNOPRSTVXZ[]\\abcdefgilmnoprstuy"))
        self.id_to_token = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]

    def encode(self, smiles: str) -> List[int]:
        ids = [self.bos_token_id]
        for char in smiles:
            ids.append(self.token_to_id.get(char, self.eos_token_id))
        ids.append(self.eos_token_id)
        return ids

    def batch_encode(self, smiles_batch: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(smiles) for smiles in smiles_batch]
        max_len = max(len(row) for row in encoded) if encoded else 0
        input_ids = torch.full((len(encoded), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for row_idx, row in enumerate(encoded):
            input_ids[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_idx, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FallbackMolEncoder(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        self.encoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        encoded = self.proj(encoded)
        encoded = encoded * attention_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return encoded


class MolLlamaWrapper(nn.Module):
    def __init__(
        self,
        model_path: str | Path | None,
        *,
        latent_dim: int = 256,
        trust_remote_code: bool = True,
        allow_remote_weights: bool = False,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.model_path = str(model_path or "")
        self.allow_remote_weights = bool(allow_remote_weights)
        self.trust_remote_code = bool(trust_remote_code)
        self.freeze_backbone = bool(freeze_backbone)

        self.tokenizer = None
        self.backbone = None
        self.output_proj = None
        self.mode = "fallback"

        if _TRANSFORMERS_OK and self.model_path:
            path = Path(self.model_path)
            tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
            model_kwargs = {"trust_remote_code": self.trust_remote_code}
            if path.exists():
                tokenizer_kwargs["local_files_only"] = True
                model_kwargs["local_files_only"] = True
            elif not self.allow_remote_weights:
                tokenizer_kwargs["local_files_only"] = True
                model_kwargs["local_files_only"] = True
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tokenizer_kwargs)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
                self.backbone = AutoModel.from_pretrained(self.model_path, **model_kwargs)
                hidden_size = int(getattr(self.backbone.config, "hidden_size", self.latent_dim))
                self.output_proj = nn.Linear(hidden_size, self.latent_dim)
                self.mode = "hf"
                if self.freeze_backbone:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
            except Exception:
                self.tokenizer = None
                self.backbone = None
                self.output_proj = None
                self.mode = "fallback"

        if self.mode == "fallback":
            self.tokenizer = _FallbackTokenizer()
            self.backbone = _FallbackMolEncoder(len(self.tokenizer.id_to_token), self.latent_dim)
            self.output_proj = nn.Identity()

    def _batch_tokenize(self, smiles_batch: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
        if self.mode == "hf":
            encoded = self.tokenizer(
                list(smiles_batch),
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer.batch_encode(list(smiles_batch))
        return {key: value.to(device) for key, value in encoded.items()}

    def _pool_sequence(self, sequence: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.to(dtype=sequence.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (sequence * weights).sum(dim=1) / denom

    def encode(self, smiles_batch: Sequence[str], chirality_signatures: Sequence[torch.Tensor] | None = None) -> List[LatentBlueprint]:
        if not smiles_batch:
            return []
        device = next(self.parameters()).device
        batch_inputs = self._batch_tokenize(smiles_batch, device)
        if self.mode == "hf":
            outputs = self.backbone(**batch_inputs, output_hidden_states=True)
            sequence = outputs.last_hidden_state
            if self.output_proj is not None:
                sequence = self.output_proj(sequence)
        else:
            sequence = self.backbone(batch_inputs["input_ids"], batch_inputs["attention_mask"])
        pooled = self._pool_sequence(sequence, batch_inputs["attention_mask"])

        blueprints: List[LatentBlueprint] = []
        batch_size = int(sequence.shape[0])
        for idx in range(batch_size):
            mask = batch_inputs["attention_mask"][idx].bool()
            seq_i = sequence[idx][mask]
            ids_i = batch_inputs["input_ids"][idx][mask]
            chirality = (
                chirality_signatures[idx].to(device)
                if chirality_signatures is not None
                else torch.zeros(8, dtype=sequence.dtype, device=device)
            )
            blueprints.append(
                LatentBlueprint(
                    sequence=seq_i,
                    pooled=pooled[idx],
                    token_ids=ids_i,
                    attention_mask=batch_inputs["attention_mask"][idx][mask],
                    smiles=str(smiles_batch[idx]),
                    source=self.mode,
                    chirality_signature=chirality,
                )
            )
        return blueprints

    def encode_one(self, smiles: str, chirality_signature: torch.Tensor | None = None) -> LatentBlueprint:
        signatures = [chirality_signature] if chirality_signature is not None else None
        return self.encode([smiles], chirality_signatures=signatures)[0]
