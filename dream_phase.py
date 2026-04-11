"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     DREAM PHASE FOR SoM PREDICTION                                           ║
║                                                                              ║
║     Inspired by biological sleep and cognitive consolidation:                ║
║                                                                              ║
║     1. REPLAY - Re-process difficult/important examples                      ║
║     2. IMAGINATION - Generate synthetic molecules in latent space            ║
║     3. CONSOLIDATION - Cluster memory, extract symbolic rules                ║
║     4. REFINEMENT - Prune/merge enzyme states                               ║
║                                                                              ║
║     "To sleep, perchance to dream—ay, there's the rub,                       ║
║      For in that sleep what patterns may emerge..."                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DreamMemory:
    """Storage for experiences to replay during dreaming."""
    embeddings: List[torch.Tensor] = field(default_factory=list)
    som_masks: List[torch.Tensor] = field(default_factory=list)
    predictions: List[torch.Tensor] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    difficulties: List[float] = field(default_factory=list)  # How hard was this example?
    
    max_size: int = 1000
    
    def add(
        self,
        embedding: torch.Tensor,
        som_mask: torch.Tensor,
        prediction: torch.Tensor,
        loss: float,
    ):
        """Add an experience to dream memory."""
        # Compute difficulty = loss * (1 - correct)
        pred_idx = prediction.argmax()
        true_idx = som_mask.argmax() if som_mask.sum() > 0 else -1
        correct = float(pred_idx == true_idx)
        difficulty = loss * (1 - correct + 0.1)  # Hard examples have high difficulty
        
        self.embeddings.append(embedding.detach().cpu())
        self.som_masks.append(som_mask.detach().cpu())
        self.predictions.append(prediction.detach().cpu())
        self.losses.append(loss)
        self.difficulties.append(difficulty)
        
        # Keep only max_size, prioritizing difficult examples
        if len(self.embeddings) > self.max_size:
            self._prune()
    
    def _prune(self):
        """Remove easiest examples to stay under max_size."""
        if len(self.embeddings) <= self.max_size:
            return
        
        # Sort by difficulty (descending) and keep top max_size
        indices = sorted(
            range(len(self.difficulties)),
            key=lambda i: self.difficulties[i],
            reverse=True
        )[:self.max_size]
        
        self.embeddings = [self.embeddings[i] for i in indices]
        self.som_masks = [self.som_masks[i] for i in indices]
        self.predictions = [self.predictions[i] for i in indices]
        self.losses = [self.losses[i] for i in indices]
        self.difficulties = [self.difficulties[i] for i in indices]
    
    def sample_batch(self, batch_size: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of experiences, weighted by difficulty."""
        if len(self.embeddings) < batch_size:
            return None
        
        # Sample weighted by difficulty
        total_diff = sum(self.difficulties)
        if total_diff == 0:
            weights = [1.0 / len(self.difficulties)] * len(self.difficulties)
        else:
            weights = [d / total_diff for d in self.difficulties]
        
        indices = random.choices(range(len(self.embeddings)), weights=weights, k=batch_size)
        
        return {
            'embeddings': torch.stack([self.embeddings[i] for i in indices]).to(device),
            'som_masks': torch.stack([self.som_masks[i] for i in indices]).to(device),
            'predictions': torch.stack([self.predictions[i] for i in indices]).to(device),
        }
    
    def __len__(self):
        return len(self.embeddings)


class DreamPhase(nn.Module):
    """
    Dream phase that runs between epochs to consolidate learning.
    
    Components:
    1. Replay: Re-process difficult examples with noise
    2. Imagination: Generate synthetic examples via interpolation
    3. Consolidation: Cluster and extract rules from memory
    4. Refinement: Optimize enzyme state representations
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        n_replay_steps: int = 10,
        n_imagination_steps: int = 5,
        noise_scale: float = 0.1,
        imagination_temperature: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_replay_steps = n_replay_steps
        self.n_imagination_steps = n_imagination_steps
        self.noise_scale = noise_scale
        self.imagination_temperature = imagination_temperature
        
        # Dream memory
        self.dream_memory = DreamMemory()
        
        # Imagination network (VAE-like)
        self.imagination_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # mu and logvar
        )
        
        self.imagination_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Rule extraction network
        self.rule_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 16),  # 16 "rule" dimensions
        )
        
        # Rule bank (learned symbolic patterns)
        self.register_buffer('rule_bank', torch.zeros(32, 16))  # 32 rules
        self.register_buffer('rule_counts', torch.zeros(32))
        self.register_buffer('rule_som_patterns', torch.zeros(32, 200))  # SoM patterns per rule
        
        # Consolidation statistics
        self.n_dreams = 0
        self.total_replay_loss = 0.0
        self.total_imagination_loss = 0.0
        self.rules_discovered = 0
    
    def record_experience(
        self,
        embedding: torch.Tensor,
        som_mask: torch.Tensor,
        prediction: torch.Tensor,
        loss: float,
    ):
        """Record an experience for later dreaming."""
        self.dream_memory.add(embedding, som_mask, prediction, loss)
    
    def _replay_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Replay difficult examples with noise injection.
        
        The noise simulates "fuzzy" recall during dreaming and builds robustness.
        """
        embeddings = batch['embeddings']  # [B, hidden_dim]
        som_masks = batch['som_masks']    # [B, N]
        predictions = batch['predictions']  # [B, N]
        
        B = embeddings.shape[0]
        N = som_masks.shape[1]
        
        # Add noise to embeddings (dream distortion)
        noise = torch.randn_like(embeddings) * self.noise_scale
        noisy_embeddings = embeddings + noise
        
        # Create per-atom features by combining global embedding with position encoding
        # This gives each atom a unique representation
        position_encoding = torch.zeros(B, N, self.hidden_dim, device=embeddings.device)
        for i in range(N):
            # Simple sinusoidal position encoding
            position_encoding[:, i, :] = torch.sin(
                torch.arange(self.hidden_dim, device=embeddings.device).float() * (i + 1) / 100
            )
        
        # Combine global embedding with position
        atom_features = noisy_embeddings.unsqueeze(1) + 0.1 * position_encoding
        
        # Add noise to predictions (counterfactual targets)
        pred_noise = torch.randn_like(predictions) * 0.1
        noisy_targets = (predictions.softmax(dim=-1) + pred_noise).clamp(min=1e-8)
        noisy_targets = noisy_targets / noisy_targets.sum(dim=-1, keepdim=True)
        
        # Also use actual SoM as soft target
        som_normalized = som_masks / (som_masks.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Mix: 70% actual SoM, 30% noisy prediction
        target = 0.7 * som_normalized + 0.3 * noisy_targets
        target = target / (target.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Forward through fusion head
        if hasattr(model, 'fusion'):
            memory_hints = torch.zeros(B, N, device=embeddings.device)
            scores, physics_scores, _ = model.fusion(atom_features, memory_hints)
            
            # Replay loss with label smoothing
            log_probs = F.log_softmax(scores, dim=-1).clamp(min=-100)
            loss = -(target * log_probs).sum(dim=-1).mean()
            
            # Add consistency loss: physics should match final
            physics_log_probs = F.log_softmax(physics_scores, dim=-1).clamp(min=-100)
            consistency = F.kl_div(
                physics_log_probs,
                scores.softmax(dim=-1).detach(),
                reduction='batchmean'
            ).clamp(max=5.0)
            
            loss = loss + 0.1 * consistency
            
            if torch.isfinite(loss) and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                return loss.item()
        
        return 0.0
    
    def _imagination_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Generate synthetic "imagined" molecules by interpolating in latent space.
        
        This explores the manifold between known examples, potentially discovering
        new patterns that help generalization.
        """
        embeddings = batch['embeddings']  # [B, hidden_dim]
        som_masks = batch['som_masks']    # [B, N]
        
        B = embeddings.shape[0]
        N = som_masks.shape[1]
        
        if B < 2:
            return 0.0
        
        # Encode to latent (VAE-style)
        encoded = self.imagination_encoder(embeddings)
        mu, logvar = encoded.chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 10)  # Prevent extreme values
        
        # Sample with reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * self.imagination_temperature
        z = mu + eps * std
        
        # Interpolate between random pairs (imagine hybrid molecules)
        idx1 = torch.randperm(B, device=embeddings.device)
        idx2 = torch.randperm(B, device=embeddings.device)
        alpha = torch.rand(B, 1, device=embeddings.device)
        
        z_interp = alpha * z[idx1] + (1 - alpha) * z[idx2]
        
        # Interpolate SoM targets too
        som_interp = alpha * som_masks[idx1] + (1 - alpha) * som_masks[idx2]
        som_interp = som_interp / (som_interp.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Decode imagined embeddings
        imagined = self.imagination_decoder(z_interp)
        
        # Create per-atom features with position encoding
        position_encoding = torch.zeros(B, N, self.hidden_dim, device=embeddings.device)
        for i in range(N):
            position_encoding[:, i, :] = torch.sin(
                torch.arange(self.hidden_dim, device=embeddings.device).float() * (i + 1) / 100
            )
        
        atom_features = imagined.unsqueeze(1) + 0.1 * position_encoding
        
        # Score imagined molecules
        if hasattr(model, 'fusion'):
            memory_hints = torch.zeros(B, N, device=embeddings.device)
            scores, _, _ = model.fusion(atom_features, memory_hints)
            
            # Imagination loss: predicted should match interpolated target
            log_probs = F.log_softmax(scores, dim=-1).clamp(min=-100)
            recon_loss = -(som_interp * log_probs).sum(dim=-1).mean()
            
            # KL divergence for VAE regularization (encourage diverse latents)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss.clamp(max=10.0)
            
            # Smoothness loss: nearby latents should have similar predictions
            z_diff = (z[idx1] - z[idx2]).norm(dim=-1, keepdim=True)
            som_diff = (som_masks[idx1] - som_masks[idx2]).abs().sum(dim=-1, keepdim=True)
            smoothness = (z_diff * som_diff).mean()
            
            loss = recon_loss + 0.01 * kl_loss + 0.01 * smoothness
            
            if torch.isfinite(loss) and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                return loss.item()
        
        return 0.0
    
    def _consolidation_step(
        self,
        model: nn.Module,
    ):
        """
        Consolidate memory by clustering and extracting rules.
        
        This is the "insight" phase of dreaming - finding patterns across examples.
        """
        if len(self.dream_memory) < 32:
            return
        
        device = next(model.parameters()).device
        
        # Get all embeddings
        all_emb = torch.stack(self.dream_memory.embeddings).to(device)
        all_som = torch.stack(self.dream_memory.som_masks).to(device)
        
        # Pad or truncate SoM to 200
        N_som = all_som.shape[1]
        if N_som < 200:
            all_som = F.pad(all_som, (0, 200 - N_som))
        elif N_som > 200:
            all_som = all_som[:, :200]
        
        # Extract rules
        with torch.no_grad():
            rules = self.rule_extractor(all_emb)  # [N, 16]
            rules_norm = F.normalize(rules, dim=-1)
        
        # Cluster by similarity to rule bank
        rule_bank_norm = F.normalize(self.rule_bank + 1e-8, dim=-1)
        similarities = torch.mm(rules_norm, rule_bank_norm.t())  # [N, 32]
        
        # Assign each example to best matching rule
        best_rules = similarities.argmax(dim=-1)  # [N]
        
        # Update rule bank with running average
        for rule_idx in range(32):
            mask = (best_rules == rule_idx)
            if mask.sum() > 0:
                # Update rule embedding
                new_rule = rules_norm[mask].mean(dim=0)
                self.rule_bank[rule_idx] = 0.9 * self.rule_bank[rule_idx] + 0.1 * new_rule
                
                # Update rule count
                self.rule_counts[rule_idx] += mask.sum()
                
                # Update SoM pattern for this rule
                som_pattern = all_som[mask].mean(dim=0)
                self.rule_som_patterns[rule_idx] = (
                    0.9 * self.rule_som_patterns[rule_idx] + 0.1 * som_pattern
                )
        
        # Count discovered rules (those with significant usage)
        self.rules_discovered = (self.rule_counts > 10).sum().item()
    
    def _refinement_step(
        self,
        model: nn.Module,
    ):
        """
        Refine enzyme state representations.
        
        Merge similar states, prune unused ones, sharpen distinctions.
        """
        if not hasattr(model, 'state_bank'):
            return
        
        state_bank = model.state_bank
        
        # Get active state embeddings
        active_states = state_bank.get_active_states()
        if active_states.shape[0] < 2:
            return
        
        # Compute pairwise similarities
        states_norm = F.normalize(active_states, dim=-1)
        sims = torch.mm(states_norm, states_norm.t())
        
        # Find highly similar pairs (excluding diagonal)
        sims.fill_diagonal_(0)
        max_sim = sims.max()
        
        # If states are too similar (>0.95), push them apart
        if max_sim > 0.95:
            # Find the pair
            idx = (sims == max_sim).nonzero()[0]
            i, j = idx[0].item(), idx[1].item()
            
            # Get actual indices in state bank
            active_indices = state_bank.state_active.nonzero(as_tuple=True)[0]
            real_i, real_j = active_indices[i], active_indices[j]
            
            # Push apart by adding noise in orthogonal direction
            with torch.no_grad():
                diff = state_bank.state_embeddings[real_i] - state_bank.state_embeddings[real_j]
                diff_norm = F.normalize(diff.unsqueeze(0), dim=-1).squeeze()
                
                # Add orthogonal noise
                noise = torch.randn_like(diff)
                noise = noise - (noise @ diff_norm) * diff_norm  # Make orthogonal
                noise = F.normalize(noise.unsqueeze(0), dim=-1).squeeze() * 0.1
                
                state_bank.state_embeddings[real_i] += noise
                state_bank.state_embeddings[real_j] -= noise
    
    def dream(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Run a complete dream cycle.
        
        This should be called between epochs (or every few epochs).
        """
        model.train()
        self.n_dreams += 1
        
        metrics = {
            'replay_loss': 0.0,
            'imagination_loss': 0.0,
            'rules_discovered': 0,
            'n_memories': len(self.dream_memory),
        }
        
        if len(self.dream_memory) < 16:
            return metrics
        
        # Phase 1: Replay
        replay_losses = []
        for _ in range(self.n_replay_steps):
            batch = self.dream_memory.sample_batch(16, device)
            if batch is not None:
                loss = self._replay_step(model, batch, optimizer)
                replay_losses.append(loss)
        
        if replay_losses:
            metrics['replay_loss'] = sum(replay_losses) / len(replay_losses)
        
        # Phase 2: Imagination
        imagination_losses = []
        for _ in range(self.n_imagination_steps):
            batch = self.dream_memory.sample_batch(16, device)
            if batch is not None:
                loss = self._imagination_step(model, batch, optimizer)
                imagination_losses.append(loss)
        
        if imagination_losses:
            metrics['imagination_loss'] = sum(imagination_losses) / len(imagination_losses)
        
        # Phase 3: Consolidation
        self._consolidation_step(model)
        metrics['rules_discovered'] = self.rules_discovered
        
        # Phase 4: Refinement
        self._refinement_step(model)
        
        return metrics
    
    def get_rule_hints(
        self,
        embedding: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get SoM hints based on discovered rules.
        
        This allows the model to use consolidated knowledge during inference.
        """
        if self.rules_discovered == 0:
            return torch.zeros(200, device=device)
        
        # Extract rule for this embedding
        with torch.no_grad():
            rule = self.rule_extractor(embedding.unsqueeze(0).to(device))
            rule_norm = F.normalize(rule, dim=-1)
            
            # Find best matching rule
            rule_bank_norm = F.normalize(self.rule_bank, dim=-1)
            sims = torch.mm(rule_norm, rule_bank_norm.t()).squeeze()
            
            # Weighted combination of rule patterns
            weights = F.softmax(sims / 0.1, dim=-1)
            hint = (weights.unsqueeze(-1) * self.rule_som_patterns).sum(dim=0)
        
        return hint


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def dream_enhanced_training_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    dream_phase: DreamPhase,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    pocket_features: torch.Tensor,
    device: torch.device,
    record_dreams: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Training step that records experiences for dreaming.
    """
    model.train()
    
    features = batch['features'].to(device)
    coords = batch['coords'].to(device)
    som_mask = batch['som_mask'].to(device)
    valid_mask = batch['valid_mask'].to(device)
    
    optimizer.zero_grad()
    
    outputs = model(
        features,
        coords,
        pocket_features.to(device),
        som_mask,
        valid_mask,
    )
    
    loss, metrics = loss_fn(outputs, som_mask, valid_mask)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Record experiences for dreaming
    if record_dreams and hasattr(model, 'mol_encoder'):
        with torch.no_grad():
            mol_encoded = model.mol_encoder(features)
            mol_global = mol_encoded.mean(dim=1)
            
            # Record each example in the batch
            B = features.shape[0]
            for b in range(B):
                dream_phase.record_experience(
                    embedding=mol_global[b],
                    som_mask=som_mask[b],
                    prediction=outputs['final_scores'][b],
                    loss=metrics['main_loss'],
                )
    
    return loss.item(), metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Dream Phase...")
    
    from hybrid_nexus_dynamic import HybridNexusDynamic, HybridLoss
    
    # Create model and dream phase
    model = HybridNexusDynamic(mol_dim=128, hidden_dim=64, max_states=32)
    dream = DreamPhase(hidden_dim=64)
    loss_fn = HybridLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(dream.parameters()),
        lr=0.001
    )
    
    device = torch.device('cpu')
    model.to(device)
    dream.to(device)
    
    # Simulate training and recording experiences
    print("Recording experiences...")
    for i in range(100):
        B, N = 4, 30
        features = torch.randn(B, N, 128)
        coords = torch.randn(B, N, 3) * 10
        pocket = torch.randn(100, 14)
        som_mask = torch.zeros(B, N)
        for b in range(B):
            som_mask[b, (b + i) % N] = 1
        valid_mask = torch.ones(B, N, dtype=torch.bool)
        
        model.train()
        outputs = model(features, coords, pocket, som_mask, valid_mask)
        loss, _ = loss_fn(outputs, som_mask, valid_mask)
        
        # Record
        mol_encoded = model.mol_encoder(features)
        mol_global = mol_encoded.mean(dim=1)
        for b in range(B):
            dream.record_experience(
                mol_global[b], som_mask[b], outputs['final_scores'][b], loss.item()
            )
    
    print(f"Dream memory size: {len(dream.dream_memory)}")
    
    # Run dream phase
    print("\nDreaming...")
    metrics = dream.dream(model, optimizer, device)
    print(f"Dream metrics: {metrics}")
    
    # Test rule hints
    test_emb = torch.randn(64)
    hints = dream.get_rule_hints(test_emb, device)
    print(f"Rule hints shape: {hints.shape}")
    print(f"Rules discovered: {dream.rules_discovered}")
    
    print("\n✓ Dream Phase test passed!")
