import torch
import torch.nn as nn
import numpy as np
from config import Config

class FISLoss(nn.Module):
    """
    Fair Identity Scaling (FIS) Loss
    
    Based on:
    - FairVision: Equitable Deep Learning for Eye Disease Screening (arXiv:2310.02492)
    - Fair Distillation: Teaching Fairness from Biased Teachers (arXiv:2411.11939)
    
    Combines individual-level and group-level scaling to improve fairness:
    - Individual Scaling: Prioritizes hard samples using softmax over losses
    - Group Scaling: Prioritizes underrepresented groups using Optimal Transport distances
    """
    
    def __init__(self, base_loss_fn=None, c=0.5, eps=Config.fis_params['eps']):
        """
        Args:
            base_loss_fn: Base loss function (default: CrossEntropyLoss)
            c: Fusion weight between individual (1-c) and group (c) scaling
               Paper uses c=0.5 for balanced trade-off
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn if base_loss_fn else nn.CrossEntropyLoss(reduction='none')
        self.c = c  # Fixed at 0.5 in papers
        self.eps = eps

    def forward(self, outputs, targets, groups):
        """
        Compute FIS loss with individual and group scaling
        
        Args:
            outputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            groups: Group assignments [batch_size]
        
        Returns:
            Scalar loss value
        """
        groups = groups.to(outputs.device)
        
        # Compute instance-level losses (Equation 3)
        losses = self.base_loss_fn(outputs, targets)  # [batch_size]
        
        # Individual Scaling (Equation 4): s_I_i = exp(l_i) / Σ exp(l_j)
        # This is softmax over losses - samples with higher losses get higher weights
        s_I = self._compute_individual_scaling(losses)
        
        # Group Scaling (Equation 5): s_G_i = exp(OT(...)) / Σ exp(OT(...))
        # Uses Optimal Transport distance between batch and group distributions
        s_G = self._compute_group_scaling(groups, losses)
        
        # Combine individual and group scaling (Equation 3)
        # L_FIS = Σ [(1-c) * s_I_i + c * s_G_i] * l_i
        final_weights = (1 - self.c) * s_I + self.c * s_G
        
        # Weighted loss
        weighted_losses = final_weights * losses
        return weighted_losses.mean()
    
    def _compute_individual_scaling(self, losses):
        """
        Individual Scaling (Equation 4 from paper)
        
        Formula: s_I_i = exp(l_i) / Σ_j exp(l_j)
        
        This is softmax normalization over losses, which:
        - Gives higher weights to samples with higher losses (hard samples)
        - Ensures weights sum to 1 across the batch
        """
        # Apply softmax to losses (using F.softmax for numerical stability)
        s_I = torch.nn.functional.softmax(losses, dim=0)
        return s_I
    
    def _compute_group_scaling(self, groups, losses):
        """
        Group Scaling (Equation 5 from paper)
        
        Formula: s_G_i = exp(OT({l}|B, {l}|B_ai)) / Σ_j exp(OT({l}|B, {l}|B_aj))
        
        Where OT is the Optimal Transport (Wasserstein) distance between:
        - The full batch loss distribution {l}|B
        - The specific group's loss distribution {l}|B_ai
        
        Groups with more different loss distributions get higher weights.
        """
        unique_groups = torch.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups == 1:
            # Only one group - uniform weights
            return torch.ones_like(losses, dtype=torch.float32) / len(losses)
        
        # Compute Optimal Transport distance for each group
        ot_distances = []
        
        # Keep everything in PyTorch to maintain gradients
        for g in unique_groups:
            group_mask = (groups == g)
            group_losses = losses[group_mask]
            
            # Compute Wasserstein distance between batch and group distributions
            ot_dist = self._wasserstein_distance_1d(losses, group_losses)
            ot_distances.append(ot_dist)
        
        # Apply softmax to OT distances (Equation 5)
        ot_tensor = torch.stack(ot_distances)
        ot_weights = torch.nn.functional.softmax(ot_tensor, dim=0)
        
        # Assign group weights to each sample based on their group
        s_G = torch.zeros_like(losses, dtype=torch.float32)
        for idx, g in enumerate(unique_groups):
            group_mask = (groups == g)
            s_G[group_mask] = ot_weights[idx]
        
        return s_G
    
    def _wasserstein_distance_1d(self, dist1, dist2):
        """
        Compute 1D Wasserstein (Earth Mover's) distance between two distributions
        
        For 1D distributions, the Wasserstein distance has a closed form:
        W(P, Q) = integral of |F_P(x) - F_Q(x)| dx
        
        Which simplifies to the mean absolute difference between sorted values.
        
        Args:
            dist1: First distribution (tensor)
            dist2: Second distribution (tensor)
        
        Returns:
            Wasserstein distance (scalar tensor)
        """
        # Handle edge cases
        if len(dist1) == 0 or len(dist2) == 0:
            return torch.tensor(0.0, device=dist1.device)
        
        # Sort both distributions
        sorted_dist1, _ = torch.sort(dist1)
        sorted_dist2, _ = torch.sort(dist2)
        
        # If distributions have different sizes, interpolate to match
        len1, len2 = len(sorted_dist1), len(sorted_dist2)
        
        if len1 != len2:
            # Use the larger size for interpolation
            target_len = max(len1, len2)
            
            # Helper function for linear interpolation in PyTorch
            def interpolate(source, target_length):
                source_len = len(source)
                if source_len == target_length:
                    return source
                
                # Create grid for interpolation
                # We want to map indices [0, source_len-1] to [0, target_length-1]
                # source_indices = torch.arange(source_len, dtype=torch.float32, device=source.device)
                target_indices = torch.linspace(0, source_len - 1, target_length, device=source.device)
                
                # Get floor and ceil indices
                idx_floor = torch.floor(target_indices).long()
                idx_ceil = torch.ceil(target_indices).long()
                
                # Clip indices to be safe
                idx_floor = torch.clamp(idx_floor, 0, source_len - 1)
                idx_ceil = torch.clamp(idx_ceil, 0, source_len - 1)
                
                # Get values
                val_floor = source[idx_floor]
                val_ceil = source[idx_ceil]
                
                # Calculate weights
                weight = target_indices - idx_floor.float()
                
                # Interpolate
                return val_floor * (1 - weight) + val_ceil * weight

            if len1 < target_len:
                sorted_dist1 = interpolate(sorted_dist1, target_len)
            
            if len2 < target_len:
                sorted_dist2 = interpolate(sorted_dist2, target_len)
        
        # Wasserstein distance = mean absolute difference of CDFs
        return torch.mean(torch.abs(sorted_dist1 - sorted_dist2))