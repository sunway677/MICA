import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import sys
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def calculate_metrics(label, pred):
    acc = calculate_acc(label, pred)
    # nmi = v_measure_score(label, pred)
    nmi = normalized_mutual_info_score(label, pred)
    pur = calculate_purity(label, pred)
    ari = adjusted_rand_score(label, pred)

    return acc, nmi, pur, ari


def calculate_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind_row, ind_col = linear_sum_assignment(w.max() - w)

    # u = linear_sum_assignment(w.max() - w)
    # ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size

# def calculate_acc(true_labels, pred_labels):
#     # 构建混淆矩阵
#     C = confusion_matrix(true_labels, pred_labels)
#
#     # 匈牙利算法找到最优匹配
#     row_ind, col_ind = linear_sum_assignment(-7)  # 寻找最大值
#
#     # 计算准确率
#     accuracy = C[row_ind, col_ind].sum() / np.sum(7)
#
#     return accuracy
def calculate_purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster_index in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster_index] = winner

    return accuracy_score(y_true, y_voted_labels)

def compute_joint(x_out, x_tf_out):
# produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency by DCP (2022TPAMI)"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))
    
    loss = loss.sum()

    return loss

# def compute_similarity(A, B, mask_A, mask_B):
#     # A: features[vi]
#     # B: features[vj]
#     # overlap_idx: the indeices of samples that exist in both A and B
#     bool_mask_A = mask_A > 0
#     bool_mask_B = mask_B > 0
#     # Get overlapping indices
#     overlap_idx = torch.nonzero(bool_mask_A & bool_mask_B).squeeze()
#
#     # Get A-unique indices
#     A_unique_idx = torch.nonzero(bool_mask_A & ~bool_mask_B).squeeze()
#
#     # Get B-unique indices
#     B_unique_idx = torch.nonzero(~bool_mask_A & bool_mask_B).squeeze()
#
#     # overlap_idx_true = (mask_A == 1) & (mask_B == 1)
#     # overlap_idx = torch.nonzero(overlap_idx_true).squeeze(dim=-1)
#
#
#     if overlap_idx.size() == 1:
#        overlap_idx = overlap_idx.unsqueeze(0)
#     A_overlap = A[overlap_idx]
#     B_overlap = B[overlap_idx]
#     sim_matrix_overlap = (torch.mm(A_overlap, B_overlap.T) / 1).sum() / (2*overlap_idx.size(0))
#
#     # Handle unique samples
#     A_unique = A[A_unique_idx]
#     B_unique = B[B_unique_idx]
#     if A_unique_idx.numel() == 0 or B_unique_idx.numel() == 0:
#         sim_matrix_unique = 0
#     else:
#         if A_unique.dim() == 1:
#             A_unique = A_unique.unsqueeze(0)
#         if B_unique.dim() == 1:
#             B_unique = B_unique.unsqueeze(0)
#         sim_matrix_unique = (torch.mm(A_unique, B_unique.T) / 1).sum() / (A_unique.size(0)+B_unique.size(0))
#
#     # Combine the similarity measures
#     combined_size = A_unique.size(0) + B_unique.size(0) + A_overlap.size(0)
#
#     combined_similarity = (sim_matrix_overlap + sim_matrix_unique) / combined_size
#
#     return sim_matrix_overlap.item()

# def compute_similarity(A, B, mask_A, mask_B):
#     # A: features[vi]
#     # B: features[vj]
#     # mask_A: mask for view A
#     # mask_B: mask for view B
#
#     bool_mask_A = mask_A > 0
#     bool_mask_B = mask_B > 0
#
#     # Get overlapping indices
#     overlap_idx = torch.nonzero(bool_mask_A & bool_mask_B).squeeze()
#
#     # Get A-unique indices
#     A_unique_idx = torch.nonzero(bool_mask_A & ~bool_mask_B).squeeze()
#
#     # Get B-unique indices
#     B_unique_idx = torch.nonzero(~bool_mask_A & bool_mask_B).squeeze()
#
#     # Handle the case when there's only one overlapping sample
#     if overlap_idx.dim() == 0:
#         overlap_idx = overlap_idx.unsqueeze(0)
#
#     A_overlap = A[overlap_idx] if overlap_idx.numel() > 0 else torch.empty(0, A.size(1))
#     B_overlap = B[overlap_idx] if overlap_idx.numel() > 0 else torch.empty(0, B.size(1))
#
#     if A_overlap.size(0) > 0 and B_overlap.size(0) > 0:
#         sim_matrix_overlap = (torch.mm(A_overlap, B_overlap.T) / 1).sum() / (2 * overlap_idx.size(0))
#     else:
#         sim_matrix_overlap = torch.tensor(0.0)
#
#     # Handle unique samples
#     if A_unique_idx.numel() == 0 or B_unique_idx.numel() == 0:
#         sim_matrix_unique = torch.tensor(0.0)
#         A_unique = torch.empty(0, A.size(1))
#         B_unique = torch.empty(0, B.size(1))
#     else:
#         A_unique = A[A_unique_idx]
#         B_unique = B[B_unique_idx]
#         if A_unique.dim() == 1:
#             A_unique = A_unique.unsqueeze(0)
#         if B_unique.dim() == 1:
#             B_unique = B_unique.unsqueeze(0)
#         sim_matrix_unique = (torch.mm(A_unique, B_unique.T) / 1).sum() / (A_unique.size(0) + B_unique.size(0))
#
#     # Combine the similarity measures
#     combined_size = A_unique.size(0) + B_unique.size(0) + A_overlap.size(0)
#     if combined_size > 0:
#         combined_similarity = (sim_matrix_overlap * A_overlap.size(0) + sim_matrix_unique * (
#                     A_unique.size(0) + B_unique.size(0))) / combined_size
#     else:
#         combined_similarity = torch.tensor(0.0)
#
#     return combined_similarity.item()
#
# def knn_indices_cosine(matrix, j, k):
#     # Calculate cosine similarity between sample j and all other samples
#     sample_j = matrix[j].unsqueeze(0)  # Add batch dimension
#     cosine_similarities = torch.nn.functional.cosine_similarity(sample_j, matrix, dim=1)
#
#     # Convert cosine similarity to cosine distance (1 - cosine similarity)
#     #cosine_distances = 1 - cosine_similarities
#
#     # Sort the distances and get indices of the K nearest neighbors
#     knn_cosine_similarities, knn_indices = torch.topk(cosine_similarities, k, largest=True)
#
#     return knn_cosine_similarities, knn_indices

# Example usage:
# matrix is your tensor of shape (N, D), where N is the number of samples and D is the dimensionality
# j is the index of the sample you want to find nearest neighbors for
# k is the number of nearest neighbors to find
# knn_indices = knn_indices_cosine(matrix, j, k)

