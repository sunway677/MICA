"""
Loss Functions for Multi-View Contrastive Learning
This module implements specialized loss functions for multi-view contrastive learning and clustering tasks.
Key components include distribution target calculation, cross-view consistency measurement,
and contrastive loss computations at both cluster and instance levels.
"""

import time
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.functional import normalize
from sklearn.neighbors import NearestNeighbors

# Import custom modules
from metrics import *
from dataprocessing import *


# Commented out PolyLoss implementation - can be uncommented if needed
# class PolyLoss(nn.Module):
#     def __init__(self, lambda_poly=1.0):
#         """
#         Implementation of Poly Loss: a more robust loss for classification tasks
#
#         Args:
#             lambda_poly (float): Weight for the polynomial term
#         """
#         super(PolyLoss, self).__init__()
#         self.lambda_poly = lambda_poly
#
#     def forward(self, logits, targets):
#         # Calculate cross-entropy loss
#         ce_loss = F.cross_entropy(logits, targets, reduction='none')
#
#         # Get predicted probability distribution
#         probs = F.softmax(logits, dim=-1)
#
#         # Convert targets to one-hot encoding
#         target_one_hot = F.one_hot(targets, num_classes=logits.size(-1))
#
#         # Calculate polynomial loss term
#         poly_loss = torch.sum(target_one_hot * (1 - probs), dim=-1)
#
#         # Calculate total loss
#         total_loss = ce_loss + self.lambda_poly * poly_loss
#
#         return total_loss.mean()


class DeepMVCLoss(nn.Module):
    """
    Deep Multi-View Clustering Loss implementation
    Combines contrastive learning approach with clustering objectives
    """

    def __init__(self, num_samples, num_clusters, lambda_poly=1.0):
        """
        Initialize the Deep Multi-View Clustering Loss module

        Args:
            num_samples (int): Number of samples in each batch
            num_clusters (int): Number of clusters in the dataset
            lambda_poly (float): Weight factor for polynomial loss term
        """
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters

        # Initialize similarity measure and loss criterion
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        # self.criterion = PolyLoss(lambda_poly)  # Uncomment to use PolyLoss

    def mask_correlated_samples(self, N):
        """
        Create a mask for correlated samples to exclude them from negative pairs

        Args:
            N (int): Number of samples

        Returns:
            torch.Tensor: Boolean mask indicating which samples can be used as negative pairs
        """
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)  # Exclude self-pairs

        # Exclude positive pairs (cross-view correspondences)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    def forward_prob(self, q_i):
        """
        Calculate entropy of target distribution

        Args:
            q_i (torch.Tensor): Probability distribution

        Returns:
            torch.Tensor: Entropy value
        """

        def entropy(q):
            # Normalize to ensure probability distribution sums to 1
            p = q.sum(0) / q.sum()
            return (p * torch.log(p)).sum()  # Calculate entropy

        q_i_target = self.target_distribution(q_i)
        # print(entropy(q_i_target))
        return entropy(q_i_target)

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        """
        Calculate contrastive loss between probability distributions at both cluster and instance levels

        Args:
            q_i (torch.Tensor): Probability distribution from first view
            q_j (torch.Tensor): Probability distribution from second view
            temperature_l (float): Temperature parameter to control distribution sharpness
            normalized (bool): Whether to use normalized similarity calculation

        Returns:
            torch.Tensor: Combined contrastive loss
        """
        # Cluster-level contrastive learning
        # Step 1: Prepare probability distributions
        # Using raw distributions instead of target distribution
        # q_i_target = self.target_distribution(q_i).t()
        # q_j_target = self.target_distribution(q_j).t()
        q_combined = torch.cat((q_i.t(), q_j.t()), dim=0)  # Combined tensor for efficiency

        # Step 2: Calculate similarity matrix
        if normalized:
            # Use cosine similarity for normalized features
            sim_matrix = self.similarity(q_combined.unsqueeze(1), q_combined.unsqueeze(0)) / temperature_l
        else:
            # Use dot product for unnormalized features
            sim_matrix = torch.matmul(q_combined, q_combined.T) / temperature_l

        # Steps 3 & 4: Extract positive and negative pairs' similarities
        # Positive pairs are the corresponding clusters across views
        pos_sim_i_j = torch.diag(sim_matrix, self.num_clusters)
        pos_sim_j_i = torch.diag(sim_matrix, -self.num_clusters)
        positive_similarities = torch.cat((pos_sim_i_j, pos_sim_j_i)).view(-1, 1)

        # Negative pairs are all other combinations excluding self-pairs
        mask = self.mask_correlated_samples(2 * self.num_clusters)
        negative_similarities = sim_matrix[mask].view(2 * self.num_clusters, -1)

        # Step 5: Calculate Loss (InfoNCE formulation)
        logits = torch.cat((positive_similarities, negative_similarities), dim=1)
        labels = torch.zeros(2 * self.num_clusters, dtype=torch.long, device=logits.device)
        ce_loss = self.criterion(logits, labels)

        # Additional polynomial loss term for robustness
        pt = torch.exp(-ce_loss)
        poly_loss = ce_loss + 1.0 * (1 - pt)

        # Combine both loss components
        loss = ce_loss / (2 * self.num_clusters) + poly_loss / (2 * self.num_clusters)

        # Instance-level contrastive learning
        batch_size = q_i.size(0)
        q_combined_sample = torch.cat((q_i, q_j), dim=0)
        sim_matrix_sample = torch.matmul(q_combined_sample, q_combined_sample.T) / temperature_l

        # Steps 6: Extract positive and negative pairs' similarities at instance level
        pos_sim_i_j_sample = torch.diag(sim_matrix_sample, batch_size)
        pos_sim_j_i_sample = torch.diag(sim_matrix_sample, -batch_size)
        positive_similarities_sample = torch.cat((pos_sim_i_j_sample, pos_sim_j_i_sample)).view(-1, 1)

        mask_sample = self.mask_correlated_samples(2 * batch_size)
        negative_similarities_sample = sim_matrix_sample[mask_sample].view(2 * batch_size, -1)

        # Step 7: Calculate instance-level loss and combine with cluster-level loss
        logits_sample = torch.cat((positive_similarities_sample, negative_similarities_sample), dim=1)
        labels_sample = torch.zeros(2 * batch_size, dtype=torch.long, device=logits.device)
        ce_loss_sample = self.criterion(logits_sample, labels_sample)

        # Add instance-level loss to the total loss
        loss = loss + ce_loss_sample / (batch_size)

        return loss

    def cross_view_consistency_loss(self, encoded_features):
        """
        Calculate consistency loss between different view encodings

        Args:
            encoded_features (list): List of feature tensors from different views

        Returns:
            torch.Tensor: Consistency loss between view pairs
        """
        num_views = len(encoded_features)
        consistency_loss = 0.0

        # Calculate MSE loss between all pairs of views
        for i in range(num_views):
            for j in range(i + 1, num_views):
                consistency_loss += torch.nn.functional.mse_loss(encoded_features[i], encoded_features[j])

        # Normalize by the number of comparisons
        consistency_loss /= (num_views * (num_views - 1) / 2)
        return consistency_loss

    def target_distribution(self, q):
        """
        Calculate target distribution for self-training in clustering
        This is a key component for deep clustering methods

        Args:
            q (torch.Tensor): Initial soft cluster assignment probabilities

        Returns:
            torch.Tensor: Target distribution with higher confidence assignments
        """
        # Formula: p_ij = (q_ij^2 / sum_i(q_ij)) / sum_j(q_ij^2 / sum_i(q_ij))
        # This sharpens the distribution and normalizes it
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p