"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     HAMILTONIAN DYNAMICS LOSS                                                ║
║                                                                              ║
║     Key insight: Learning STATE is not enough - we must learn DYNAMICS      ║
║                                                                              ║
║     Standard MSE loss:                                                       ║
║       L = ||pred_state - true_state||²                                       ║
║       Problem: Model learns to predict MEAN (flat lines)                     ║
║                                                                              ║
║     Dynamic loss:                                                            ║
║       L = L_state + λ₁·L_delta + λ₂·L_direction + λ₃·L_energy               ║
║       - L_delta: ||pred_change - true_change||²                             ║
║       - L_direction: sign(pred_change) == sign(true_change)                 ║
║       - L_energy: Hamiltonian should be conserved                           ║
║                                                                              ║
║     This forces the model to learn actual dynamics, not just averages.      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HamiltonianDynamicsLoss(nn.Module):
    """
    Loss function that enforces learning of DYNAMICS, not just states.
    
    The key problem with standard MSE:
    - If 90% of metabolites are constant, model learns "predict constant"
    - The 10% that change get ignored
    - Result: good loss, but no dynamics learned
    
    Solution - weight loss by CHANGE:
    - Penalize wrong deltas heavily
    - Penalize wrong direction (up vs down)
    - Reward energy conservation (Hamiltonian)
    """
    
    def __init__(
        self,
        delta_weight: float = 10.0,      # Weight for delta (change) loss
        direction_weight: float = 5.0,   # Weight for direction loss
        energy_weight: float = 1.0,      # Weight for energy conservation
        variance_weight: float = 2.0,    # Weight for variance-based reweighting
        use_variance_weighting: bool = True,
    ):
        super().__init__()
        
        self.delta_weight = delta_weight
        self.direction_weight = direction_weight
        self.energy_weight = energy_weight
        self.variance_weight = variance_weight
        self.use_variance_weighting = use_variance_weighting
    
    def compute_state_loss(
        self,
        pred: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Standard state prediction loss.
        
        Args:
            pred: (B, D) predicted state
            target: (B, D) true state
            weights: (D,) optional per-dimension weights
        """
        error = (pred - target) ** 2
        
        if weights is not None:
            error = error * weights.unsqueeze(0)
        
        return error.mean()
    
    def compute_delta_loss(
        self,
        pred_next: Tensor,
        true_next: Tensor,
        current: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Loss on the CHANGE (delta) - this is the key innovation.
        
        Forces model to predict dynamics, not just mean.
        
        Args:
            pred_next: (B, D) predicted next state
            true_next: (B, D) true next state
            current: (B, D) current state
            weights: (D,) optional per-dimension weights
        """
        pred_delta = pred_next - current
        true_delta = true_next - current
        
        error = (pred_delta - true_delta) ** 2
        
        if weights is not None:
            error = error * weights.unsqueeze(0)
        
        return error.mean()
    
    def compute_direction_loss(
        self,
        pred_next: Tensor,
        true_next: Tensor,
        current: Tensor,
    ) -> Tensor:
        """
        Loss for getting direction (up/down) right.
        
        Even if magnitude is wrong, at least get the sign right.
        """
        pred_delta = pred_next - current
        true_delta = true_next - current
        
        # Sign agreement: +1 if same sign, -1 if different
        pred_sign = torch.sign(pred_delta)
        true_sign = torch.sign(true_delta)
        
        # Penalize when signs disagree (and true change is significant)
        # Use smooth approximation for differentiability
        agreement = pred_sign * true_sign  # +1 if agree, -1 if disagree
        
        # Only penalize significant changes
        significant = true_delta.abs() > 0.001
        
        # Loss: want agreement to be +1, so minimize (1 - agreement)
        direction_error = F.relu(1 - agreement) * significant.float()
        
        return direction_error.mean()
    
    def compute_energy_loss(
        self,
        energies: Tensor,
    ) -> Tensor:
        """
        Energy conservation loss.
        
        For Hamiltonian dynamics, H should be conserved.
        Large drift = model not respecting physics.
        
        Args:
            energies: (T,) or (T, B) energy at each timestep
        """
        if energies.dim() == 1:
            energies = energies.unsqueeze(-1)
        
        # Energy should be constant
        energy_var = energies.var(dim=0)
        
        # Normalize by initial energy magnitude
        initial_energy = energies[0].abs().clamp(min=1e-6)
        relative_drift = energy_var / initial_energy
        
        return relative_drift.mean()
    
    def compute_variance_weights(
        self,
        trajectory: Tensor,
    ) -> Tensor:
        """
        Compute per-dimension weights based on variance.
        
        Dimensions that change more should matter more in the loss.
        
        Args:
            trajectory: (T, B, D) or (B, T, D) trajectory
        """
        if trajectory.dim() == 3:
            # Compute variance over time
            variance = trajectory.var(dim=0 if trajectory.shape[0] < trajectory.shape[1] else 1)
            variance = variance.mean(dim=0)  # Average over batch -> (D,)
        else:
            variance = trajectory.var(dim=0)
        
        # Add small constant to avoid division by zero
        variance = variance + 1e-6
        
        # Weight = sqrt(variance) / mean(sqrt(variance))
        # This upweights dimensions that vary more
        weights = variance.sqrt()
        weights = weights / weights.mean()
        
        return weights.clamp(min=0.1, max=10.0)
    
    def forward(
        self,
        pred_trajectory: Tensor,      # (T, B, D) or (B, T, D)
        true_trajectory: Tensor,      # Same shape
        energies: Optional[Tensor] = None,  # (T,) or (T, B)
        time_first: bool = True,      # Whether time is dim 0 or 1
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined dynamic loss.
        
        Args:
            pred_trajectory: Predicted trajectory
            true_trajectory: True trajectory
            energies: Optional Hamiltonian energy at each timestep
            time_first: Whether first dim is time (True) or batch (False)
            
        Returns:
            loss: Total loss
            metrics: Dict of individual loss components
        """
        if not time_first:
            pred_trajectory = pred_trajectory.transpose(0, 1)
            true_trajectory = true_trajectory.transpose(0, 1)
        
        T = pred_trajectory.shape[0]
        
        # Compute variance-based weights if enabled
        weights = None
        if self.use_variance_weighting:
            weights = self.compute_variance_weights(true_trajectory)
        
        # Aggregate losses over timesteps
        state_loss = 0.0
        delta_loss = 0.0
        direction_loss = 0.0
        
        for t in range(1, T):
            pred_next = pred_trajectory[t]
            true_next = true_trajectory[t]
            current = true_trajectory[t - 1]
            
            state_loss += self.compute_state_loss(pred_next, true_next, weights)
            delta_loss += self.compute_delta_loss(pred_next, true_next, current, weights)
            direction_loss += self.compute_direction_loss(pred_next, true_next, current)
        
        # Normalize by number of steps
        n_steps = T - 1
        state_loss = state_loss / n_steps
        delta_loss = delta_loss / n_steps
        direction_loss = direction_loss / n_steps
        
        # Energy loss
        energy_loss = torch.tensor(0.0, device=pred_trajectory.device)
        if energies is not None:
            energy_loss = self.compute_energy_loss(energies)
        
        # Total loss
        total_loss = (
            state_loss
            + self.delta_weight * delta_loss
            + self.direction_weight * direction_loss
            + self.energy_weight * energy_loss
        )
        
        metrics = {
            'state_loss': state_loss.item(),
            'delta_loss': delta_loss.item(),
            'direction_loss': direction_loss.item(),
            'energy_loss': energy_loss.item() if isinstance(energy_loss, Tensor) else energy_loss,
            'total_loss': total_loss.item(),
        }
        
        return total_loss, metrics


class TrajectoryPredictionLoss(nn.Module):
    """
    Simplified loss for trajectory prediction.
    
    Use this when you don't have full Hamiltonian energies,
    just predicted vs true trajectories.
    """
    
    def __init__(
        self,
        delta_weight: float = 10.0,
        direction_weight: float = 5.0,
    ):
        super().__init__()
        self.delta_weight = delta_weight
        self.direction_weight = direction_weight
    
    def forward(
        self,
        pred_next: Tensor,    # (B, D)
        true_next: Tensor,    # (B, D)
        true_curr: Tensor,    # (B, D)
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Single-step loss.
        """
        # State loss
        state_loss = F.mse_loss(pred_next, true_next)
        
        # Delta loss
        pred_delta = pred_next - true_curr
        true_delta = true_next - true_curr
        delta_loss = F.mse_loss(pred_delta, true_delta)
        
        # Direction loss (smooth)
        pred_sign = torch.tanh(pred_delta * 10)  # Soft sign
        true_sign = torch.tanh(true_delta * 10)
        direction_loss = F.mse_loss(pred_sign, true_sign)
        
        total_loss = (
            state_loss
            + self.delta_weight * delta_loss
            + self.direction_weight * direction_loss
        )
        
        metrics = {
            'state_loss': state_loss.item(),
            'delta_loss': delta_loss.item(),
            'direction_loss': direction_loss.item(),
        }
        
        return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def dynamic_train_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    loss_fn: HamiltonianDynamicsLoss,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Single training step with dynamic loss.
    
    Example usage:
        loss, metrics = dynamic_train_step(model, batch, loss_fn, device)
        loss.backward()
        optimizer.step()
    """
    gene_traj = batch['gene_trajectory'].to(device)
    met_traj = batch['met_trajectory'].to(device)
    
    B, T, _ = gene_traj.shape
    
    # Collect predictions
    pred_trajectory = [met_traj[:, 0]]
    energies = []
    
    current_met = met_traj[:, 0]
    
    for t in range(T - 1):
        out = model(gene_traj[:, t], current_met)
        pred_next = out['next_metabolites']
        pred_trajectory.append(pred_next)
        
        if 'energies' in out:
            energies.append(out['energies'][-1])
        
        # Teacher forcing: use true state for next step
        current_met = met_traj[:, t + 1]
    
    pred_trajectory = torch.stack(pred_trajectory, dim=1)  # (B, T, D)
    
    energies = torch.stack(energies) if energies else None
    
    loss, metrics = loss_fn(
        pred_trajectory.transpose(0, 1),  # (T, B, D)
        met_traj.transpose(0, 1),
        energies=energies,
    )
    
    return loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing HamiltonianDynamicsLoss...")
    
    # Create test data
    T, B, D = 20, 4, 50
    
    # True trajectory with clear trends
    trends = torch.randn(B, D) * 0.05
    true_traj = torch.zeros(T, B, D)
    true_traj[0] = torch.randn(B, D)
    for t in range(1, T):
        true_traj[t] = true_traj[t-1] + trends + 0.01 * torch.randn(B, D)
    
    # Predicted trajectory (model trying to learn)
    # Bad model: predicts flat
    pred_flat = true_traj[0].unsqueeze(0).expand(T, -1, -1).clone()
    
    # Good model: captures dynamics
    pred_good = true_traj + 0.1 * torch.randn_like(true_traj)
    
    # Energy (should be conserved)
    energies = torch.ones(T) * 10 + 0.1 * torch.randn(T)
    
    # Test loss
    loss_fn = HamiltonianDynamicsLoss()
    
    loss_flat, metrics_flat = loss_fn(pred_flat, true_traj, energies)
    loss_good, metrics_good = loss_fn(pred_good, true_traj, energies)
    
    print(f"\nFlat prediction (bad):")
    print(f"  Total loss: {loss_flat.item():.4f}")
    print(f"  Metrics: {metrics_flat}")
    
    print(f"\nGood prediction:")
    print(f"  Total loss: {loss_good.item():.4f}")
    print(f"  Metrics: {metrics_good}")
    
    print(f"\nGood model loss should be << flat model loss:")
    print(f"  {loss_good.item():.4f} << {loss_flat.item():.4f}")
    print(f"  Ratio: {loss_good.item() / loss_flat.item():.4f}")
    
    if loss_good < loss_flat:
        print("\n✓ Dynamic loss correctly penalizes flat predictions!")
    else:
        print("\n✗ Something wrong - flat should be penalized more")
