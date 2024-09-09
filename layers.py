import torch.nn as nn
import torch.nn.functional as F
from metrics import *



class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, dims: list, dropout_p: float = 0.5):
        super(AutoEncoder, self).__init__()
        self.encoder = self._build_layers(input_dim, feature_dim, dims, dropout_p)

    def _build_layers(self, input_dim, feature_dim, dims, dropout_p):
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
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, dims: list, dropout_p: float = 0.5):
        super(AutoDecoder, self).__init__()
        self.decoder = self._build_layers(feature_dim, input_dim, list(reversed(dims)), dropout_p)

    def _build_layers(self, input_dim, feature_dim, dims, dropout_p):
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
        return self.decoder(x)


def knn_indices_cosine(matrix, j, k):
    # Calculate cosine similarity between sample j and all other samples
    sample_j = matrix[j].unsqueeze(0)  # Add batch dimension
    cosine_similarities = torch.nn.functional.cosine_similarity(sample_j, matrix, dim=1)

    # Convert cosine similarity to cosine distance (1 - cosine similarity)
    # cosine_distances = 1 - cosine_similarities

    # Sort the distances and get indices of the K nearest neighbors
    knn_cosine_similarities, knn_indices = torch.topk(cosine_similarities, k, largest=True)

    return knn_cosine_similarities, knn_indices


def compute_similarity(A, B, mask_A, mask_B):
    # A: features[vi]
    # B: features[vj]
    # mask_A: mask for view A
    # mask_B: mask for view B

    bool_mask_A = mask_A > 0
    bool_mask_B = mask_B > 0

    # Get overlapping indices
    overlap_idx = torch.nonzero(bool_mask_A & bool_mask_B).squeeze()

    # Get A-unique indices
    A_unique_idx = torch.nonzero(bool_mask_A & ~bool_mask_B).squeeze()

    # Get B-unique indices
    B_unique_idx = torch.nonzero(~bool_mask_A & bool_mask_B).squeeze()

    # Handle the case when there's only one overlapping sample
    if overlap_idx.dim() == 0:
        overlap_idx = overlap_idx.unsqueeze(0)

    A_overlap = A[overlap_idx] if overlap_idx.numel() > 0 else torch.empty(0, A.size(1))
    B_overlap = B[overlap_idx] if overlap_idx.numel() > 0 else torch.empty(0, B.size(1))

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
        A_unique = A[A_unique_idx]
        B_unique = B[B_unique_idx]
        if A_unique.dim() == 1:
            A_unique = A_unique.unsqueeze(0)
        if B_unique.dim() == 1:
            B_unique = B_unique.unsqueeze(0)
        sim_matrix_unique = (torch.mm(A_unique, B_unique.T) / 1).sum() / (A_unique.size(0) + B_unique.size(0))

    # Combine the similarity measures
    combined_size = A_unique.size(0) + B_unique.size(0) + A_overlap.size(0)
    if combined_size > 0:
        combined_similarity = (sim_matrix_overlap * A_overlap.size(0) + sim_matrix_unique * (
                A_unique.size(0) + B_unique.size(0))) / combined_size
    else:
        combined_similarity = torch.tensor(0.0)

    return combined_similarity.item()


class CVCLNetwork(nn.Module):
    def __init__(self, num_views: int, input_sizes: list, dims: list, dim_high_feature: int, dim_low_feature: int,
                 num_clusters: int, batch_size: int):
        super(CVCLNetwork, self).__init__()
        self.encoders = nn.ModuleList(
            [AutoEncoder(input_sizes[idx], dim_high_feature, dims) for idx in range(num_views)])
        self.decoders = nn.ModuleList(
            [AutoDecoder(input_sizes[idx], dim_high_feature, dims) for idx in range(num_views)])
        self.encode_features = [nn.Parameter(torch.zeros(batch_size, dim_high_feature)) for _ in range(num_views)]

        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, data_views: list, missing_info: torch.Tensor, phase_code: bool):
        lbps = []
        dvs = []
        features = []
        data_views_new = data_views

        num_views = len(data_views)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature extraction and reconstruction
        for idx in range(num_views):
            data_view = data_views[idx].to(device).float()
            high_features = self.encoders[idx](data_view)
            data_view_recon = self.decoders[idx](high_features)
            dvs.append(data_view_recon.clone())
            features.append(high_features.clone())

        dvs_new = dvs

        # Missing-feature imputation and inference
        if phase_code:
            mutual_info_graph = torch.zeros(num_views, num_views)
            for vi in range(num_views):
                for vj in range(num_views):
                    #if vi != vj:
                    mutual_info_graph[vi, vj] = compute_similarity(features[vi], features[vj], 1-missing_info[:, vi], 1-missing_info[:, vj])
                    # if vi != vj:
                    #     overlap_idx_true = (missing_info[:, vi] == 0) & (missing_info[:, vj] == 0)
                    #     overlap_idx = torch.nonzero(overlap_idx_true).squeeze(dim=-1)
                    #     if overlap_idx.size() == 1:
                    #         overlap_idx = overlap_idx.unsqueeze(0)
                    #     mutual_info_graph[vi, vj] = compute_similarity(features[vi], features[vj], overlap_idx)
                        #mutual_info_graph[vi, vj] = -instance_contrastive_Loss(features[vi][overlap_idx],
            # Normalize each row by its corresponding diagonal element
            for vi in range(num_views):
                dia = mutual_info_graph[vi, vi]
                if dia != 0:  # Ensure no division by zero
                    mutual_info_graph[vi, :] = mutual_info_graph[vi, :] / dia
                mutual_info_graph[vi, vi] = 0

            #print(mutual_info_graph)
            missing_info_imputing = missing_info

            for vi in range(num_views):
                missing_idx = torch.nonzero(missing_info_imputing[:, vi] == 1).squeeze()
                if missing_idx.dim() != 0:
                    for j in missing_idx:
                        view_existing_idx = (missing_info_imputing[j, :] == 0)
                        graph_idx = mutual_info_graph[vi].to(device)
                        view_existing_idx = view_existing_idx.to(device)
                        max_idx = torch.argmax(torch.mul(graph_idx.clone(), view_existing_idx.clone()))
                        max_view_weight = graph_idx[max_idx]
                        k=20
                        cosine_similarities, knn_indices = knn_indices_cosine(features[max_idx].clone().detach(), j, k)
                        cosine_similarities = torch.clamp(cosine_similarities, min=0)
                        missing_set = set(missing_idx.tolist())
                        knn_set = set(knn_indices.tolist())
                        unique_knn_indices = knn_set - missing_set
                        unique_knn_indices_tensor = torch.tensor(list(unique_knn_indices))
                        if len(unique_knn_indices_tensor) > 0:
                            indices_in_knn_set = [list(knn_set).index(idx) for idx in unique_knn_indices]
                            knn_cosine_similarities = cosine_similarities[indices_in_knn_set]
                            knn_features = features[vi][unique_knn_indices_tensor].clone()
                            weighted_knn_features = knn_features * knn_cosine_similarities.unsqueeze(1)
                            features[vi][j] = max_view_weight * torch.sum(weighted_knn_features, dim=0) / torch.sum(
                                knn_cosine_similarities)

                            knn_input = data_views_new[vi][unique_knn_indices_tensor].clone()
                            weighted_knn_input = knn_input * knn_cosine_similarities.unsqueeze(1)
                            data_views_new[vi] = data_views_new[vi].clone()
                            data_views_new[vi][j] = max_view_weight * torch.sum(weighted_knn_input, dim=0) / torch.sum(
                                knn_cosine_similarities)

                            knn_out = dvs[vi][unique_knn_indices_tensor].clone()
                            weighted_knn_out = knn_out * knn_cosine_similarities.unsqueeze(1)
                            dvs_new[vi][j] = max_view_weight * torch.sum(weighted_knn_out, dim=0) / torch.sum(knn_cosine_similarities)

                label_probs = self.label_learning_module(features[vi].clone())
                lbps.append(label_probs)

        fused_features = torch.mean(torch.stack(features), dim=0)
        #fused_probs = self.label_learning_module(fused_features.clone())

        new_features = []
        input_feature_loss = 0
        output_feature_loss = 0
        for idx in range(num_views):
            data_view_new = data_views_new[idx].to(device).float()
            high_features_new = self.encoders[idx](data_view_new)
            new_features.append(high_features_new.clone())
            input_feature_loss += F.mse_loss(high_features_new, features[idx])

            data_view_recon_new = self.decoders[idx](features[idx])
            output_feature_loss += F.mse_loss(data_view_recon_new, dvs_new[idx])

        return lbps, dvs, fused_features, features, input_feature_loss, output_feature_loss




    # No imputation
    # def forward(self, data_views: list, missing_info: torch.Tensor, phase_code: bool):
    #     lbps = []
    #     dvs = []
    #     features = []
    #     data_views_new = data_views
    #
    #     num_views = len(data_views)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     # Feature extraction and reconstruction
    #     for idx in range(num_views):
    #         data_view = data_views[idx].to(device).float()
    #         high_features = self.encoders[idx](data_view)
    #         data_view_recon = self.decoders[idx](high_features)
    #         dvs.append(data_view_recon.clone())
    #         features.append(high_features.clone())
    #
    #         fused_features = torch.mean(torch.stack(features), dim=0)
    #         fused_probs = self.label_learning_module(fused_features.clone())
    #         label_probs = self.label_learning_module(features[idx].clone())
    #         lbps.append(label_probs)
    #         new_features = []
    #         input_feature_loss = 0
    #         output_feature_loss = 0
    #         for idx in range(num_views):
    #             data_view_new = data_views_new[idx].to(device).float()
    #             high_features_new = self.encoders[idx](data_view_new)
    #             new_features.append(high_features_new.clone())
    #
    #
    #     return lbps, dvs, fused_probs, features, input_feature_loss, output_feature_loss


