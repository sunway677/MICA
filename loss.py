import time
import torch
import torch.nn as nn
import numpy as np
import math
from metrics import *
import torch.nn.functional as F
from torch.nn.functional import normalize
from dataprocessing import *
from sklearn.neighbors import NearestNeighbors


class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters, k=5):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.k = k   # 近邻的数量

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    def forward_prob(self, q_i, q_j):
        def entropy(q):
            p = q.sum(0) / q.sum()  # 归一化，确保概率分布的和为1
            return (p * torch.log(p)).sum()  # 计算熵

        q_i_target = self.target_distribution(q_i)
        q_j_target = self.target_distribution(q_j)

        return entropy(q_i_target) + entropy(q_j_target)

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        # Step 1: Process through target_distribution
        q_i_target = self.target_distribution(q_i).t()
        q_j_target = self.target_distribution(q_j).t()
        q_combined = torch.cat((q_i_target, q_j_target), dim=0)  # Combined tensor for simplicity

        # Step 2: Calculate similarity matrix
        if normalized:
            sim_matrix = self.similarity(q_combined.unsqueeze(1), q_combined.unsqueeze(0)) / temperature_l
        else:
            sim_matrix = torch.matmul(q_combined, q_combined.T) / temperature_l

        # Steps 3 & 4: Extract positive and negative pairs' similarities
        pos_sim_i_j = torch.diag(sim_matrix, self.num_clusters)
        pos_sim_j_i = torch.diag(sim_matrix, -self.num_clusters)
        positive_similarities = torch.cat((pos_sim_i_j, pos_sim_j_i)).view(-1, 1)

        mask = self.mask_correlated_samples(2 * self.num_clusters)
        negative_similarities = sim_matrix[mask].view(2 * self.num_clusters, -1)

        # Step 5: Calculate Loss
        logits = torch.cat((positive_similarities, negative_similarities), dim=1)
        labels = torch.zeros(2 * self.num_clusters, dtype=torch.long, device=logits.device)
        loss = self.criterion(logits, labels) / (2 * self.num_clusters)

        return loss

    # Cross-View Consistency Loss
    def cross_view_consistency_loss(self, encoded_features):
        num_views = len(encoded_features)
        consistency_loss = 0.0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                consistency_loss += torch.nn.functional.mse_loss(encoded_features[i], encoded_features[j])
        # Normalize by the number of comparisons
        consistency_loss /= (num_views * (num_views - 1) / 2)
        return consistency_loss

    # def fusion_loss(self, fused_feature, original_features):
    #     fusion_loss = 0
    #     for feature in original_features:
    #         # 计算fused_feature与每个视图特征之间的MSE损失
    #         fusion_loss += F.mse_loss(fused_feature, feature)
    #     fusion_loss /= len(original_features)
    #     return fusion_loss
    def fusion_loss(self, fused_feature, original_features, tau=0.01):
        similarities = [F.cosine_similarity(fused_feature.unsqueeze(0), feature.unsqueeze(0), dim=1) for feature in
                        original_features]
        similarities = torch.stack(similarities)

        # 使用Softmax函数转换相似度为权重
        weights = F.softmax(similarities / tau, dim=0)

        # 最大化fused feature与所有original feature加权相似度的和
        contrastive_loss = -torch.log(torch.sum(weights * similarities))

        return contrastive_loss

    def lsp_loss(self, fused_feature):
        cpu_fused_feature = fused_feature.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='auto').fit(cpu_fused_feature)
        _, indices = nbrs.kneighbors(cpu_fused_feature)

        # 计算LSP Loss
        lsp_loss = 0.
        for i in range(cpu_fused_feature.shape[0]):
            for j in indices[i][1:]:  # 排除自身
                lsp_loss += torch.norm(fused_feature[i] - fused_feature[j], p=2) ** 2
        lsp_loss = lsp_loss / (cpu_fused_feature.shape[0] * self.k)

        return lsp_loss

    def orth_loss(self,fused_feature):
        # 归一化特征矩阵的行
        norm_feature = torch.nn.functional.normalize(fused_feature, p=2, dim=1)

        # 计算自相关矩阵
        correlation_matrix = torch.matmul(norm_feature.transpose(0, 1), norm_feature)

        # 动态创建大小匹配的单位矩阵
        I = torch.eye(correlation_matrix.size(0)).to(fused_feature.device)

        # 计算与单位矩阵的偏差
        orth_loss = torch.norm(correlation_matrix - I, p='fro') ** 2

        return orth_loss

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        # print(q.shape)
        return (weight.t() / torch.sum(weight, 1)).t()
