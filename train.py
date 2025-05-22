"""
Training Module for Multi-View Contrastive Learning Framework
This module implements training functions for multi-view clustering with missing data imputation.
Includes pre-training, contrastive training, inference, and validation functions.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from loss import *
from metrics import *
from dataprocessing import *
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
    """
    Pre-training phase using reconstruction loss (MSE) without missing data imputation

    Args:
        network_model: The main network model
        mv_data: Multi-view dataset
        batch_size (int): Batch size for training
        epochs (int): Number of pre-training epochs
        optimizer: PyTorch optimizer

    Returns:
        np.ndarray: Array of loss values for each epoch
    """
    print("Starting pre-training phase...")
    start_time = time.time()

    # Get data loader and dataset info
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    # Initialize loss tracking
    pre_train_loss_values = np.zeros(epochs, dtype=np.float64)
    criterion = torch.nn.MSELoss()

    network_model.train()  # Set model to training mode

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (sub_data_views, _, missing_info) in enumerate(mv_data_loader):
            phase_code = False  # No imputation during pre-training

            # Forward pass through network
            _, reconstructed_views, _, _, _, _ = network_model(sub_data_views, missing_info, phase_code)

            # Calculate reconstruction loss for each view (only on non-missing data)
            view_losses = []
            for view_idx in range(num_views):
                # Create mask: 1 for available data, 0 for missing data
                availability_mask = (1 - missing_info[:, view_idx]).unsqueeze(1).expand_as(sub_data_views[view_idx])

                # Calculate MSE loss only on available data
                masked_original = sub_data_views[view_idx] * availability_mask
                masked_reconstructed = reconstructed_views[view_idx] * availability_mask
                view_loss = criterion(masked_original, masked_reconstructed)
                view_losses.append(view_loss)

            # Combine losses from all views
            total_batch_loss = sum(view_losses)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            epoch_loss += total_batch_loss.item()
            num_batches += 1

        # Record average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches
        pre_train_loss_values[epoch] = avg_epoch_loss

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Pre-training epoch {epoch}/{epochs - 1}, Average Loss: {avg_epoch_loss:.7f}')

    total_time = time.time() - start_time
    print(f"Pre-training completed in {total_time:.4f}s")

    return pre_train_loss_values


def contrastive_train(network_model, mv_data, mvc_loss, batch_size, alpha, beta, lmd, gamma, omega,
                      temperature_l, normalized, epoch, optimizer):
    """
    Contrastive training phase with missing data imputation and multi-view consistency learning

    Args:
        network_model: The main network model
        mv_data: Multi-view dataset
        mvc_loss: Multi-view contrastive loss function
        batch_size (int): Batch size for training
        alpha (float): Weight for view-specific contrastive loss
        beta (float): Weight for cross-view contrastive loss
        lmd (float): Weight for consistency and imputation loss
        gamma (float): Weight for regularization (unused in current implementation)
        omega (float): Weight for fusion loss (unused in current implementation)
        temperature_l (float): Temperature parameter for contrastive loss
        normalized (bool): Whether to use normalized similarity
        epoch (int): Current epoch number
        optimizer: PyTorch optimizer

    Returns:
        float: Total loss for this epoch
    """
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    network_model.train()

    # Get data loader and dataset info
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)

    total_epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (sub_data_views, _, missing_info) in enumerate(mv_data_loader):
        phase_code = True  # Enable imputation during contrastive training

        # Forward pass with imputation
        label_probs, reconstructed_views, fused_features, imputed_features, input_feature_loss, output_feature_loss = \
            network_model(sub_data_views, missing_info, phase_code)

        # Calculate imputation loss (using both input and output feature losses)
        imputation_loss = input_feature_loss + output_feature_loss

        # Calculate contrastive losses between different views
        contrastive_losses = []
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # Inter-view contrastive loss
                inter_view_loss = alpha * mvc_loss.forward_label(
                    label_probs[i], label_probs[j], temperature_l, normalized
                )
                contrastive_losses.append(inter_view_loss)

                # Intra-view entropy loss
                entropy_loss = beta * mvc_loss.forward_prob(label_probs[i])
                contrastive_losses.append(entropy_loss)

        # Calculate cross-view consistency loss
        consistency_loss = mvc_loss.cross_view_consistency_loss(imputed_features)

        # Combine all losses
        contrastive_loss = sum(contrastive_losses)
        total_loss = contrastive_loss + lmd * (consistency_loss + imputation_loss)

        # Add L1 regularization for sparsity
        # l1_regularization = 0.0
        # for param in network_model.parameters():
        #     l1_regularization += param.abs().sum()
        # total_loss += 0.00001 * l1_regularization

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_epoch_loss += total_loss.item()
        num_batches += 1

    # Calculate average loss for the epoch
    avg_epoch_loss = total_epoch_loss / num_batches

    # Print progress
    if epoch % 10 == 0 or epoch % 50 == 0:
        print(f'Contrastive training epoch {epoch}, Average Loss: {avg_epoch_loss:.7f}')

    return avg_epoch_loss


def inference(network_model, mv_data, batch_size):
    """
    Perform inference on the trained model to get cluster predictions

    Args:
        network_model: Trained network model
        mv_data: Multi-view dataset
        batch_size (int): Batch size for inference

    Returns:
        tuple: (predicted_labels, true_labels, fused_features_for_visualization)
    """
    print("Starting inference...")
    network_model.eval()  # Set model to evaluation mode

    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    # Storage for results
    all_true_labels = []
    all_unified_probs = []
    all_fused_features = []

    phase_code = True  # Use imputation during inference

    with torch.no_grad():  # Disable gradients for inference
        for batch_idx, (sub_data_views, sub_labels, sub_missing_info) in enumerate(mv_data_loader):
            # Forward pass
            label_probs, _, fused_features, _, _, _ = network_model(sub_data_views, sub_missing_info, phase_code)

            # Aggregate probabilities from available views
            batch_size_actual = sub_data_views[0].size(0)
            unified_probs = torch.zeros_like(label_probs[0])
            view_counts = torch.zeros((batch_size_actual, 1), device=unified_probs.device)

            # For each view, add probabilities from non-missing samples
            for view_idx in range(num_views):
                # Get availability mask (1 for available, 0 for missing)
                available_mask = (sub_missing_info[:, view_idx] == 0)

                if available_mask.any():
                    unified_probs[available_mask] += label_probs[view_idx][available_mask]
                    view_counts[available_mask] += 1

            # Average probabilities across available views
            view_counts = view_counts.clamp(min=1)  # Prevent division by zero
            unified_probs = unified_probs / view_counts

            # Store results
            all_unified_probs.append(unified_probs)
            all_true_labels.extend(sub_labels.numpy())
            all_fused_features.append(fused_features.cpu())

    # Concatenate all results
    final_unified_probs = torch.cat(all_unified_probs, dim=0)
    final_predicted_labels = torch.argmax(final_unified_probs, dim=1).cpu().numpy()
    final_fused_features = torch.cat(all_fused_features, dim=0).numpy()
    final_true_labels = np.array(all_true_labels)

    print("Inference completed.")
    return final_predicted_labels, final_true_labels, final_fused_features


def valid(network_model, mv_data, batch_size):
    """
    Validate the trained model and calculate clustering metrics

    Args:
        network_model: Trained network model
        mv_data: Multi-view dataset
        batch_size (int): Batch size for validation

    Returns:
        tuple: (accuracy, nmi, purity, ari) clustering metrics
    """
    print("Starting validation...")
    predicted_labels, true_labels, _ = inference(network_model, mv_data, batch_size)

    # Calculate clustering metrics
    print("Calculating clustering metrics...")
    acc, nmi, pur, ari = calculate_metrics(true_labels, predicted_labels)

    print("Clustering Results:")
    print(f'Accuracy (ACC) = {acc:.4f}')
    print(f'Normalized Mutual Information (NMI) = {nmi:.4f}')
    print(f'Purity (PUR) = {pur:.4f}')
    print(f'Adjusted Rand Index (ARI) = {ari:.4f}')

    return acc, nmi, pur, ari


def visualize_results_tsne(network_model, mv_data, batch_size, save_path=None):
    """
    Generate t-SNE visualization of the learned features

    Args:
        network_model: Trained network model
        mv_data: Multi-view dataset
        batch_size (int): Batch size for inference
        save_path (str, optional): Path to save the visualization

    Returns:
        tuple: (2D_features, predicted_labels, true_labels)
    """
    print("Generating t-SNE visualization...")
    predicted_labels, true_labels, fused_features = inference(network_model, mv_data, batch_size)

    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(fused_features)

    if save_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=predicted_labels, cmap='viridis', alpha=0.7, s=20)
        plt.colorbar(scatter, label='Predicted Cluster Labels')
        plt.title('t-SNE Visualization of Learned Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualization saved to {save_path}")

    return features_2d, predicted_labels, true_labels