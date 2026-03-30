from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .accessibility import AccessibilityFieldOutput, AccessibilityFieldState, NeuralImplicitAccessibilityField
from .allostery import AllostericEncoderOutput
from .attention import ReversedGeometricAttention
from .dynamics import DynamicPocketSimulator, DynamicPocketState
from .encoder import PocketEncoderOutput, SEGNNPocketEncoder
from .hypernetwork import IsoformHyperOutput, IsoformSpecificHyperNetwork
from .nftm import NFTMReadout
from .pga import PGA_DIM, geometric_inner_product
from nexus.physics.pga_math import pga_geometric_product


@dataclass
class EnzymePocketEncodingOutput:
    gated_multivector_field: torch.Tensor
    refined_coords: torch.Tensor
    anchors: torch.Tensor
    conditioned_field: torch.Tensor
    accessibility_mask: torch.Tensor
    accessibility_state: AccessibilityFieldState
    accessibility_output: AccessibilityFieldOutput
    nftm_readout: NFTMReadout
    pocket: PocketEncoderOutput
    hyper: IsoformHyperOutput
    dynamic_state: DynamicPocketState
    attention_weights: torch.Tensor
    attn_scores: torch.Tensor


class NFTMController(nn.Module):
    def __init__(
        self,
        residue_vocab: int = 32,
        hidden_dim: int = 128,
        algebra_dim: int = PGA_DIM,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.algebra_dim = int(algebra_dim)
        self.segnn_head = SEGNNPocketEncoder(
            residue_vocab=residue_vocab,
            hidden_dim=hidden_dim,
            layers=2,
        )
        self.read_query = nn.Linear(self.algebra_dim, hidden_dim)
        self.read_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.accessibility_field = NeuralImplicitAccessibilityField(
            residue_vocab=residue_vocab,
            hidden_dim=hidden_dim,
            memory_dim=hidden_dim,
        )
        self.a_field_writer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _infer_query_coords(drug_query_mv: torch.Tensor) -> torch.Tensor:
        if drug_query_mv.size(-1) == 8:
            return drug_query_mv[..., 1:4]   # Cl(3,0): e1,e2,e3 at indices 1-3
        if drug_query_mv.size(-1) >= 5:
            return drug_query_mv[..., 1:4]   # G(3,0,1) canonical: e1,e2,e3 at indices 1-3
        raise ValueError("drug_query_mv must have trailing dimension 8 or >=5")

    def _pad_query(self, drug_query_mv: torch.Tensor) -> torch.Tensor:
        if drug_query_mv.size(-1) == self.algebra_dim:
            return drug_query_mv
        out = drug_query_mv.new_zeros(drug_query_mv.shape[:-1] + (self.algebra_dim,))
        take = min(drug_query_mv.size(-1), self.algebra_dim)
        out[..., :take] = drug_query_mv[..., :take]
        return out

    def forward(
        self,
        protein_coords: torch.Tensor,
        drug_query_mv: torch.Tensor,
        *,
        residue_types: Optional[torch.Tensor] = None,
        conservation_scores: Optional[torch.Tensor] = None,
        isoform: Optional[IsoformHyperOutput] = None,
        allosteric: Optional[AllostericEncoderOutput] = None,
    ) -> tuple[torch.Tensor, PocketEncoderOutput, AccessibilityFieldState, AccessibilityFieldOutput, NFTMReadout]:
        squeezed = drug_query_mv.ndim == 2
        n_res = protein_coords.size(0)
        if residue_types is None:
            residue_types = torch.zeros(n_res, dtype=torch.long, device=protein_coords.device)
        pocket = self.segnn_head(
            protein_coords,
            residue_types,
            conservation_scores=conservation_scores,
        )
        state = self.accessibility_field.build_state(
            protein_coords,
            residue_types,
            pocket,
            isoform=isoform,
            allosteric=allosteric,
        )
        query_mv = self._pad_query(drug_query_mv)
        if query_mv.ndim == 2:
            query_mv = query_mv.unsqueeze(0)
        protein_features = pocket.scalar_features.unsqueeze(0).expand(query_mv.size(0), -1, -1)
        query_features = self.read_query(query_mv)
        context_vector, _ = self.read_attention(
            query=query_features,
            key=protein_features,
            value=protein_features,
        )
        coords = self._infer_query_coords(drug_query_mv)
        accessibility_output = state.query(coords)
        nftm_readout = self.accessibility_field.nftm(coords, state.nftm_state)
        a_field_mask = self.a_field_writer(context_vector + nftm_readout.memory_context.unsqueeze(0 if query_mv.ndim == 3 else 0)).squeeze(-1)
        final_mask = 0.5 * a_field_mask + 0.5 * accessibility_output.accessibility.reshape_as(a_field_mask)
        if squeezed:
            final_mask = final_mask.squeeze(0)
        return final_mask, pocket, state, accessibility_output, nftm_readout


class EnzymePocketEncoder(nn.Module):
    def __init__(
        self,
        algebra_dim: int = 16,
        embedding_dim: int = 512,
        hidden_dim: int = 128,
        residue_vocab: int = 32,
    ) -> None:
        super().__init__()
        self.algebra_dim = int(algebra_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)

        self.pocket_encoder = SEGNNPocketEncoder(
            residue_vocab=residue_vocab,
            hidden_dim=hidden_dim,
            layers=3,
        )
        self.hypernet = IsoformSpecificHyperNetwork(
            structural_dim=embedding_dim,
            allosteric_dim=embedding_dim,
            hidden_dim=hidden_dim,
            siren_hidden_dim=128,
            siren_layers=3,
        )
        self.isoform_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.a_field_controller = NFTMController(
            residue_vocab=residue_vocab,
            hidden_dim=hidden_dim,
            algebra_dim=algebra_dim,
        )
        self.dynamic_refiner = DynamicPocketSimulator(
            residue_vocab=residue_vocab,
            hidden_dim=hidden_dim,
            encoder_layers=2,
            attention_heads=4,
        )
        self.drug_transform = nn.Parameter(torch.zeros(self.algebra_dim))
        self.field_transform = nn.Parameter(torch.zeros(self.algebra_dim))
        with torch.no_grad():
            self.drug_transform[0] = 1.0
            self.field_transform[0] = 1.0
        self.reversed_attention = ReversedGeometricAttention(
            drug_dim=algebra_dim,
            pocket_dim=algebra_dim,
            heads=4,
            hidden_dim=hidden_dim,
        )

    @staticmethod
    def _infer_query_coords(drug_mv: torch.Tensor) -> torch.Tensor:
        if drug_mv.size(-1) >= 4:
            return drug_mv[..., 1:4]   # both 8D Cl(3,0) and 16D G(3,0,1): e1,e2,e3 at 1-3
        raise ValueError("drug_mv must expose at least 3 coordinate-like channels")

    def _pad_to_algebra(self, drug_mv: torch.Tensor) -> torch.Tensor:
        if drug_mv.size(-1) == self.algebra_dim:
            return drug_mv
        out = drug_mv.new_zeros(drug_mv.shape[:-1] + (self.algebra_dim,))
        take = min(drug_mv.size(-1), self.algebra_dim)
        out[..., :take] = drug_mv[..., :take]
        return out

    def forward(
        self,
        drug_mv: torch.Tensor,
        protein_coords: torch.Tensor,
        isoform_embedding: torch.Tensor,
        *,
        sequence: Optional[str] = None,
        sequence_embedding: Optional[torch.Tensor] = None,
        variant_ids: Optional[torch.Tensor] = None,
        variant_embedding: Optional[torch.Tensor] = None,
        residue_types: Optional[torch.Tensor] = None,
        conservation_scores: Optional[torch.Tensor] = None,
        allosteric: Optional[AllostericEncoderOutput] = None,
        t: Optional[torch.Tensor] = None,
    ) -> EnzymePocketEncodingOutput:
        n_res = protein_coords.size(0)
        if residue_types is None:
            residue_types = torch.zeros(n_res, dtype=torch.long, device=protein_coords.device)

        pocket = self.pocket_encoder(
            protein_coords,
            residue_types,
            conservation_scores=conservation_scores,
        )
        isoform_latent = self.isoform_projector(isoform_embedding.to(device=protein_coords.device, dtype=protein_coords.dtype).reshape(-1))
        allosteric_embedding = None if allosteric is None else allosteric.global_embedding
        hyper = self.hypernet(
            residue_positions=protein_coords,
            pocket_context=pocket.pocket_context,
            sequence=sequence,
            sequence_embedding=sequence_embedding,
            variant_ids=variant_ids,
            variant_embedding=variant_embedding,
            structural_embedding=isoform_latent,
            allosteric_embedding=allosteric_embedding,
        )

        padded_drug = self._pad_to_algebra(drug_mv)
        projected_drug = pga_geometric_product(
            padded_drug,
            self.drug_transform.view(*([1] * (padded_drug.ndim - 1)), self.algebra_dim),
        )
        attention = self.reversed_attention(projected_drug, pocket)
        anchors = pocket.attention_anchors
        if projected_drug.ndim == 2:
            attn_scores = geometric_inner_product(
                projected_drug.unsqueeze(1).expand(-1, anchors.size(0), -1),
                anchors.unsqueeze(0).expand(padded_drug.size(0), -1, -1),
            )
            attn_weights = torch.softmax(attn_scores / (float(self.algebra_dim) ** 0.5), dim=-1)
            conditioned_field = pga_geometric_product(
                attention.attended_drug,
                self.field_transform.view(1, self.algebra_dim).expand_as(attention.attended_drug),
            )
        else:
            raise ValueError(f"drug_mv must have shape [N, {self.algebra_dim}]")

        accessibility_mask, _, accessibility_state, accessibility_output, nftm_readout = self.a_field_controller(
            protein_coords,
            projected_drug,
            residue_types=residue_types,
            conservation_scores=conservation_scores,
            isoform=hyper,
            allosteric=allosteric,
        )

        query_coords = self._infer_query_coords(projected_drug)
        dynamic_state = self.dynamic_refiner.step(
            protein_coords,
            residue_types,
            projected_drug,
            query_coords,
            conservation_scores=conservation_scores,
            isoform=hyper,
            t=t,
        )
        gated_multivector_field = projected_drug * accessibility_mask.unsqueeze(-1)
        return EnzymePocketEncodingOutput(
            gated_multivector_field=gated_multivector_field,
            refined_coords=dynamic_state.residue_positions,
            anchors=anchors,
            conditioned_field=conditioned_field,
            accessibility_mask=accessibility_mask,
            accessibility_state=accessibility_state,
            accessibility_output=accessibility_output,
            nftm_readout=nftm_readout,
            pocket=pocket,
            hyper=hyper,
            dynamic_state=dynamic_state,
            attention_weights=attn_weights,
            attn_scores=attn_scores,
        )

    def gated_loss(
        self,
        pred_affinity: torch.Tensor,
        true_affinity: torch.Tensor,
        causal_adj: torch.Tensor,
        true_dag: torch.Tensor,
        a_field: torch.Tensor,
    ) -> torch.Tensor:
        loss_affinity = F.mse_loss(pred_affinity, true_affinity)
        clash_penalty = F.relu(-a_field) * 50.0
        if causal_adj.dim() == 3 and clash_penalty.dim() == 2:
            gated_causal_logits = causal_adj - clash_penalty.unsqueeze(-1)
        else:
            gated_causal_logits = causal_adj - clash_penalty
        loss_causal = F.binary_cross_entropy_with_logits(gated_causal_logits, true_dag)
        loss_steric = torch.mean(F.relu(-a_field))
        return loss_affinity + loss_causal + 0.5 * loss_steric


__all__ = [
    "EnzymePocketEncodingOutput",
    "NFTMController",
    "EnzymePocketEncoder",
]
