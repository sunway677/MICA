"""
Neural Network Architecture Components for Multi-View Contrastive Learning
This module implements the core network architecture components including:
- Encoder/Decoder networks for each view
- Feature similarity computation functions
- Main network with feature fusion and imputation capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import *


class AutoEncoder(nn.Module):
    """
    Encoder network for feature extraction from input data
    Creates a representation in a high-dimensional feature space
    """

    def __init__(self, input_dim: int, feature_dim: int, dims: list, dropout_p: float = 0.5):
        """
        Initialize the encoder network

        Args:
            input_dim (int): Dimension of input features
            feature_dim (int): Dimension of output feature space
            dims (list): List of hidden layer dimensions
            dropout_p (float): Dropout probability for regularization
        """
        super(AutoEncoder, self).__init__()
        self.encoder = self._build_layers(input_dim, feature_dim, dims, dropout_p)

    def _build_layers(self, input_dim, feature_dim, dims, dropout_p):
        """
        Build the encoder network layers

        Args:
            input_dim (int): Dimension of input features
            feature_dim (int): Dimension of output feature space
            dims (list): List of hidden layer dimensions
            dropout_p (float): Dropout probability

        Returns:
            nn.Sequential: Encoder network
        """
        layers = []
        dims = [input_dim] + dims + [feature_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.Dropout(dropout_p))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Encoded features
        """
        return self.encoder(x)


class AutoDecoder(nn.Module):
    """
    Decoder network for reconstructing input data from encoded features
    """

    def __init__(self, input_dim: int, feature_dim: int, dims: list, dropout_p: float = 0.5):
        """
        Initialize the decoder network

        Args:
            input_dim (int): Dimension of original input (to be reconstructed)
            feature_dim (int): Dimension of encoded features
            dims (list): List of hidden layer dimensions
            dropout_p (float): Dropout probability for regularization
        """
        super(AutoDecoder, self).__init__()
        self.decoder = self._build_layers(feature_dim, input_dim, list(reversed(dims)), dropout_p)

    def _build_layers(self, input_dim, feature_dim, dims, dropout_p):
        """
        Build the decoder network layers

        Args:
            input_dim (int): Dimension of encoded features (input to decoder)
            feature_dim (int): Dimension of output (reconstructed data)
            dims (list): List of hidden layer dimensions (reversed from encoder)
            dropout_p (float): Dropout probability

        Returns:
            nn.Sequential: Decoder network
        """
        layers = []
        dims = [input_dim] + dims + [feature_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.Dropout(dropout_p))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder

        Args:
            x (torch.Tensor): Encoded features

        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(x)


def knn_indices_cosine(matrix, j, k):
    """
    Find k-nearest neighbors of a sample using cosine similarity

    Args:
        matrix (torch.Tensor): Feature matrix
        j (int): Index of the target sample
        k (int): Number of nearest neighbors to find

    Returns:
        tuple: (cosine_similarities, indices) of k-nearest neighbors
    """
    # Calculate cosine similarity between sample j and all other samples
    sample_j = matrix[j].unsqueeze(0)  # Add batch dimension
    cosine_similarities = torch.nn.functional.cosine_similarity(sample_j, matrix, dim=1)

    # Sort similarities in descending order and get indices
    knn_cosine_similarities, knn_indices = torch.topk(cosine_similarities, k, largest=True)

    return knn_cosine_similarities, knn_indices


def compute_similarity(A, B, mask_A, mask_B):
    """
    Compute similarity between two sets of features considering missing data

    Args:
        A (torch.Tensor): Features from view A
        B (torch.Tensor): Features from view B
        mask_A (torch.Tensor): Mask indicating available samples in view A (1=available)
        mask_B (torch.Tensor): Mask indicating available samples in view B (1=available)

    Returns:
        float: Similarity score between the two feature sets
    """
    # Convert masks to boolean
    bool_mask_A = mask_A > 0
    bool_mask_B = mask_B > 0

    # Get different sets of indices based on availability
    overlap_idx = torch.nonzero(bool_mask_A & bool_mask_B).squeeze()  # Samples available in both views
    A_unique_idx = torch.nonzero(bool_mask_A & ~bool_mask_B).squeeze()  # Samples only in view A
    B_unique_idx = torch.nonzero(~bool_mask_A & bool_mask_B).squeeze()  # Samples only in view B

    # Handle the case when there's only one overlapping sample
    if overlap_idx.dim() == 0:
        overlap_idx = overlap_idx.unsqueeze(0)

    # Extract features for overlapping samples
    A_overlap = A[overlap_idx] if overlap_idx.numel() > 0 else torch.empty(0, A.size(1))
    B_overlap = B[overlap_idx] if overlap_idx.numel() > 0 else torch.empty(0, B.size(1))

    # Compute similarity for overlapping samples
    if A_overlap.size(0) > 0 and B_overlap.size(0) > 0:
        sim_matrix_overlap = (torch.mm(A_overlap, B_overlap.T) / 1).sum() / (2 * overlap_idx.size(0))
    else:
        sim_matrix_overlap = torch.tensor(0.0)

    # Handle unique samples
    if A_unique_idx.numel() == 0 or B_unique_idx.numel() == 0:
        sim_matrix_unique = torch.tensor(0.0)
        A_unique = torch.empty(0, A.size(1))
        B_unique = torch.empty(0, B.size(1))
    else:
        # Extract features for unique samples
        A_unique = A[A_unique_idx]
        B_unique = B[B_unique_idx]

        # Ensure tensors have correct dimensions
        if A_unique.dim() == 1:
            A_unique = A_unique.unsqueeze(0)
        if B_unique.dim() == 1:
            B_unique = B_unique.unsqueeze(0)

        # Compute similarity for unique samples
        sim_matrix_unique = (torch.mm(A_unique, B_unique.T) / 1).sum() / (A_unique.size(0) + B_unique.size(0))

    # Combine similarities with weighted averaging
    combined_size = A_unique.size(0) + B_unique.size(0) + A_overlap.size(0)
    if combined_size > 0:
        combined_similarity = (sim_matrix_overlap * A_overlap.size(0) + sim_matrix_unique * (
                A_unique.size(0) + B_unique.size(0))) / combined_size
    else:
        combined_similarity = torch.tensor(0.0)

    return combined_similarity.item()


class MainNetwork(nn.Module):
    """
    Main network architecture for multi-view contrastive learning
    Handles feature extraction, missing data imputation, and clustering
    """

    def __init__(self, num_views: int, input_sizes: list, dims: list, dim_high_feature: int, dim_low_feature: int,
                 num_clusters: int, batch_size: int):
        """
        Initialize the main network

        Args:
            num_views (int): Number of data views
            input_sizes (list): List of input dimensions for each view
            dims (list): List of hidden layer dimensions for encoder/decoder
            dim_high_feature (int): Dimension of high-level features
            dim_low_feature (int): Dimension of low-level features
            num_clusters (int): Number of clusters
            batch_size (int): Batch size for training
        """
        super(MainNetwork, self).__init__()

        # Create encoder and decoder for each view
        self.encoders = nn.ModuleList(
            [AutoEncoder(input_sizes[idx], dim_high_feature, dims) for idx in range(num_views)])
        self.decoders = nn.ModuleList(
            [AutoDecoder(input_sizes[idx], dim_high_feature, dims) for idx in range(num_views)])

        # Initialize encoded features
        self.encode_features = [nn.Parameter(torch.zeros(batch_size, dim_high_feature)) for _ in range(num_views)]

        # Cluster label prediction module
        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, data_views: list, missing_info: torch.Tensor, phase_code: bool):
        """
        Forward pass through the main network

        Args:
            data_views (list): List of data views
            missing_info (torch.Tensor): Binary tensor indicating missing data (1=missing)
            phase_code (bool): Whether to perform missing data imputation

        Returns:
            tuple: (
                lbps: List of label probabilities for each view,
                dvs: List of reconstructed data views,
                fused_features: Combined features across views,
                features: List of encoded features for each view,
                input_feature_loss: Loss between original and imputed features,
                output_feature_loss: Loss between original and reconstructed outputs
            )
        """
        lbps = []  # Label probabilities for each view
        dvs = []  # Reconstructed data views
        features = []  # Encoded features for each view
        data_views_new = data_views  # Copy of data views for imputation

        num_views = len(data_views)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Step 1: Feature extraction and reconstruction for each view
        for idx in range(num_views):
            data_view = data_views[idx].to(device).float()
            high_features = self.encoders[idx](data_view)
            data_view_recon = self.decoders[idx](high_features)
            dvs.append(data_view_recon.clone())
            features.append(high_features.clone())

        dvs_new = dvs  # Copy of reconstructed views for imputation

        # Step 2: Missing data imputation
        if phase_code:
            # Compute mutual information graph between views
            mutual_info_graph = torch.zeros(num_views, num_views)
            for vi in range(num_views):
                for vj in range(num_views):
                    mutual_info_graph[vi, vj] = compute_similarity(
                        features[vi], features[vj],
                        1 - missing_info[:, vi], 1 - missing_info[:, vj]
                    )

            # Normalize mutual information graph
            for vi in range(num_views):
                dia = mutual_info_graph[vi, vi]
                if dia != 0:  # Ensure no division by zero
                    mutual_info_graph[vi, :] = mutual_info_graph[vi, :] / dia
                mutual_info_graph[vi, vi] = 0

            missing_info_imputing = missing_info

            # Impute missing features
            for vi in range(num_views):
                # Find samples with missing features in current view
                missing_idx = torch.nonzero(missing_info_imputing[:, vi] == 1).squeeze()

                # If there are missing samples, impute them
                if missing_idx.dim() != 0:
                    for j in missing_idx:
                        # Find views where this sample is not missing
                        view_existing_idx = (missing_info_imputing[j, :] == 0)
                        graph_idx = mutual_info_graph[vi].to(device)
                        view_existing_idx = view_existing_idx.to(device)

                        # Find the most similar view with data for this sample
                        max_idx = torch.argmax(torch.mul(graph_idx.clone(), view_existing_idx.clone()))
                        max_view_weight = graph_idx[max_idx]

                        # Find k-nearest neighbors in the most similar view
                        k = 20
                        cosine_similarities, knn_indices = knn_indices_cosine(
                            features[max_idx].clone().detach(), j, k
                        )
                        cosine_similarities = torch.clamp(cosine_similarities, min=0)

                        # Remove indices that are also missing
                        missing_set = set(missing_idx.tolist())
                        knn_set = set(knn_indices.tolist())
                        unique_knn_indices = knn_set - missing_set
                        unique_knn_indices_tensor = torch.tensor(list(unique_knn_indices))

                        # If we have valid neighbors, impute using weighted average
                        if len(unique_knn_indices_tensor) > 0:
                            # Get positions in knn set
                            indices_in_knn_set = [list(knn_set).index(idx) for idx in unique_knn_indices]
                            knn_cosine_similarities = cosine_similarities[indices_in_knn_set]

                            # Impute high-level features
                            knn_features = features[vi][unique_knn_indices_tensor].clone()
                            weighted_knn_features = knn_features * knn_cosine_similarities.unsqueeze(1)
                            features[vi][j] = max_view_weight * torch.sum(weighted_knn_features, dim=0) / torch.sum(
                                knn_cosine_similarities)

                            # Impute input data
                            knn_input = data_views_new[vi][unique_knn_indices_tensor].clone()
                            weighted_knn_input = knn_input * knn_cosine_similarities.unsqueeze(1)
                            data_views_new[vi] = data_views_new[vi].clone()
                            data_views_new[vi][j] = max_view_weight * torch.sum(weighted_knn_input, dim=0) / torch.sum(
                                knn_cosine_similarities)

                            # Impute reconstructed output
                            knn_out = dvs[vi][unique_knn_indices_tensor].clone()
                            weighted_knn_out = knn_out * knn_cosine_similarities.unsqueeze(1)
                            dvs_new[vi][j] = max_view_weight * torch.sum(weighted_knn_out, dim=0) / torch.sum(
                                knn_cosine_similarities)

                # Generate label probabilities for each view
                label_probs = self.label_learning_module(features[vi].clone())
                lbps.append(label_probs)

        # Step 3: Fuse features from all views
        fused_features = torch.mean(torch.stack(features), dim=0)

        # Step 4: Apply the imputed features and calculate losses
        new_features = []
        input_feature_loss = 0
        output_feature_loss = 0

        for idx in range(num_views):
            # Process imputed input to get new features
            data_view_new = data_views_new[idx].to(device).float()
            high_features_new = self.encoders[idx](data_view_new)
            new_features.append(high_features_new.clone())

            # Calculate loss between original and imputed features
            input_feature_loss += F.mse_loss(high_features_new, features[idx])

            # Calculate loss between reconstructed outputs
            data_view_recon_new = self.decoders[idx](features[idx])
            output_feature_loss += F.mse_loss(data_view_recon_new, dvs_new[idx])

        return lbps, dvs, fused_features, features, input_feature_loss, output_feature_loss

