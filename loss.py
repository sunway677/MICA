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


# class PolyLoss(nn.Module):
#     def __init__(self, lambda_poly=1.0):
#         super(PolyLoss, self).__init__()
#         self.lambda_poly = lambda_poly
#
#     def forward(self, logits, targets):
#         # 计算交叉熵loss
#         ce_loss = F.cross_entropy(logits, targets, reduction='none')
#
#         # 获取预测的概率分布
#         probs = F.softmax(logits, dim=-1)
#
#         # one-hot
#         target_one_hot = F.one_hot(targets, num_classes=logits.size(-1))
#
#         # 计算ploy loss term
#         poly_loss = torch.sum(target_one_hot * (1 - probs), dim=-1)
#
#         # total loss
#         total_loss = ce_loss + self.lambda_poly * poly_loss
#
#         return total_loss.mean()


class DeepMVCLoss(nn.Module):
    def __init__(self, num_samples, num_clusters, lambda_poly=1.0):
        super(DeepMVCLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters

        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        # self.criterion = PolyLoss(lambda_poly)


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        return mask

    def forward_prob(self, q_i):
       
        def entropy(q):
            p = q.sum(0) / q.sum() # 归一化，确保概率分布的和为1
            return (p * torch.log(p)).sum()  # 计算熵

        q_i_target = self.target_distribution(q_i)

        # print(entropy(q_i_target))
        return entropy(q_i_target)

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        # Step 1: Process through target_distribution
        #q_i_target = self.target_distribution(q_i).t()
        #q_j_target = self.target_distribution(q_j).t()
        #q_combined = torch.cat((q_i_target, q_j_target), dim=0)  # Combined tensor for simplicity
        
        q_combined = torch.cat((q_i.t(), q_j.t()), dim=0)
        
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
        ce_loss = self.criterion(logits, labels)

        pt = torch.exp(-ce_loss)
        poly_loss = ce_loss + 1.0 * (1 - pt)

        # print(poly_loss)


        loss = ce_loss/ (2 * self.num_clusters) + poly_loss / (2 * self.num_clusters)
        
        
        ##### Instance-level
        batch_size = q_i.size(0)
        q_combined_sample = torch.cat((q_i, q_j), dim=0)
        sim_matrix_sample = torch.matmul(q_combined_sample, q_combined_sample.T) / temperature_l
        
        
        # Steps 6: Extract positive and negative pairs' similarities
        pos_sim_i_j_sample = torch.diag(sim_matrix_sample, batch_size)
        pos_sim_j_i_sample = torch.diag(sim_matrix_sample, -batch_size)
        positive_similarities_sample = torch.cat((pos_sim_i_j_sample, pos_sim_j_i_sample)).view(-1, 1)

        mask_sample = self.mask_correlated_samples(2 *batch_size)
        negative_similarities_sample = sim_matrix_sample[mask_sample].view(2 * batch_size, -1)

        # Step 7: Calculate Loss
        logits_sample = torch.cat((positive_similarities_sample, negative_similarities_sample), dim=1)
        labels_sample = torch.zeros(2 * batch_size, dtype=torch.long, device=logits.device)
        ce_loss_sample = self.criterion(logits_sample, labels_sample)
        
        loss = loss + ce_loss_sample/(batch_size)
        
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
    
    # def forward_feature(self, h_i, h_j, batch_size):
    #     N = 2 * batch_size
    #     h = torch.cat((h_i, h_j), dim=0)
    #
    #     sim = torch.matmul(h, h.T) / 0.5
    #     sim_i_j = torch.diag(sim, batch_size)
    #     sim_j_i = torch.diag(sim, -batch_size)
    #
    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    #     mask = self.mask_correlated_samples(N)
    #     negative_samples = sim[mask].reshape(N, -1)
    #
    #     labels = torch.zeros(N).to(positive_samples.device).long()
    #     logits = torch.cat((positive_samples, negative_samples), dim=1)
    #     loss = self.criterion(logits, labels)
    #     loss /= N
    #     return loss


    def target_distribution(self, q):
        #weight = (q ** 2.0) / torch.sum(q, 0)
        # print(q.shape)
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p
    
    
   
