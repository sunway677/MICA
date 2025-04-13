"""
Evaluation Metrics for Clustering Performance
This module implements various metrics to evaluate clustering quality including:
- Clustering accuracy (ACC)
- Normalized Mutual Information (NMI)
- Purity
- Adjusted Rand Index (ARI)
- Contrastive loss computations for consistency maximization
"""

import sys
import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


def calculate_metrics(label, pred):
    """
    Calculate multiple clustering performance metrics

    Args:
        label (numpy.array): Ground truth labels
        pred (numpy.array): Predicted cluster assignments

    Returns:
        tuple: (accuracy, normalized mutual information, purity, adjusted rand index)
    """
    acc = calculate_acc(label, pred)
    # Alternative implementation: nmi = v_measure_score(label, pred)
    nmi = normalized_mutual_info_score(label, pred)
    pur = calculate_purity(label, pred)
    ari = adjusted_rand_score(label, pred)

    return acc, nmi, pur, ari


def calculate_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm
    This finds the optimal one-to-one mapping between predicted and true clusters

    Args:
        y_true (numpy.array): Ground truth labels
        y_pred (numpy.array): Predicted cluster assignments

    Returns:
        float: Clustering accuracy in range [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    # Create contingency matrix (also called confusion matrix)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Find optimal one-to-one mapping using Hungarian algorithm
    # linear_sum_assignment minimizes cost, so we negate the counts
    ind_row, ind_col = linear_sum_assignment(w.max() - w)

    # Calculate accuracy based on the optimal assignment
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def calculate_purity(y_true, y_pred):
    """
    Calculate clustering purity
    Each cluster is assigned to the class which is most frequent in the cluster,
    then the accuracy of this assignment is measured

    Args:
        y_true (numpy.array): Ground truth labels
        y_pred (numpy.array): Predicted cluster assignments

    Returns:
        float: Clustering purity in range [0,1]
    """
    # Create mapping from original labels to ordered integers
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])

    # Remap original labels to ordered integers
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]

    # Create bins for histogram
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    # For each cluster, find the most common class
    for cluster_index in np.unique(y_pred):
        # Count class frequencies in this cluster
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
        # Assign the most frequent class to this cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster_index] = winner

    # Calculate accuracy of the cluster-to-class assignment
    return accuracy_score(y_true, y_voted_labels)


def compute_joint(x_out, x_tf_out):
    """
    Compute joint probability distribution between two sets of features
    Used for contrastive learning

    Args:
        x_out (torch.Tensor): First set of features
        x_tf_out (torch.Tensor): Second set of features (transformed)

    Returns:
        torch.Tensor: Joint probability distribution matrix
    """
    # Get batch size and feature dimension
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    # Compute outer product to get joint distribution
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # Sum over batch dimension to get k, k matrix

    # Symmetrize and normalize
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # Make symmetric
    p_i_j = p_i_j / p_i_j.sum()  # Normalize to ensure it's a probability distribution

    return p_i_j


def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """
    Contrastive loss for maximizing feature consistency
    Based on the Deep Comprehensive Correlation Mining (DCP) method (2022 TPAMI)

    Args:
        x_out (torch.Tensor): First set of features
        x_tf_out (torch.Tensor): Second set of features (transformed)
        lamb (float): Weight for the regularization terms
        EPS (float): Small constant to prevent numerical instability

    Returns:
        torch.Tensor: Contrastive loss value
    """
    # Get feature dimension
    _, k = x_out.size()

    # Compute joint distribution
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    # Compute marginal distributions
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)  # Marginal for rows
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # Marginal for columns

    # Avoid numerical issues with small values
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    # Compute loss based on KL divergence
    # This encourages features to be consistent while being informative
    loss = -p_i_j * (torch.log(p_i_j)
                     - lamb * torch.log(p_j)
                     - lamb * torch.log(p_i))

    # Sum over all elements
    loss = loss.sum()

    return loss