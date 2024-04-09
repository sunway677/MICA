import time

import torch
import torch.nn as nn
from loss import *
from metrics import *
from dataprocessing import *
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
    t = time.time()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    pre_train_loss_values = np.zeros(epochs, dtype=np.float64)

    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.
        for batch_idx, (sub_data_views, _, missing_info) in enumerate(mv_data_loader):
            # print(missing_info.shape)
            _, dvs, _, _ = network_model(sub_data_views, missing_info)
            loss_list = list()
            for idx in range(num_views):
                # missing_info中，1表示missing,0表示不missing的
                mask = (1 - missing_info[:, idx].unsqueeze(1)).expand_as(sub_data_views[idx])
                loss_list.append(criterion(sub_data_views[idx] * mask, dvs[idx] * mask))
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pre_train_loss_values[epoch] = total_loss
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss / num_samples))

    print("Pre-training finished.")
    print("Total time elapsed: {:.4f}s".format(time.time() - t))

    return pre_train_loss_values


def contrastive_train(network_model, mv_data, mvc_loss, batch_size, alpha, beta, lmd, gamma, omega, temperature_l,
                      normalized, epoch,
                      optimizer):
    torch.autograd.set_detect_anomaly(True)
    network_model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.
    all_encoded_features = []
    for batch_idx, (sub_data_views, _, missing_info) in enumerate(mv_data_loader):
        lbps, dvs, encoded_features, fused_features = network_model(sub_data_views)
        all_encoded_features.append(encoded_features)
        loss_list = list()
        for i in range(num_views):
            # missing_info中，1表示missing,0表示不missing的
            mask = (1 - missing_info[:, i].unsqueeze(1)).expand_as(sub_data_views[i])
            for j in range(i + 1, num_views):
                loss_list.append(alpha * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))
                loss_list.append(beta * mvc_loss.forward_prob(lbps[i], lbps[j]))
            loss_list.append(criterion(sub_data_views[i] * mask, dvs[i] * mask))

        consistency_loss = mvc_loss.cross_view_consistency_loss(encoded_features)

        # fusion_loss = mvc_loss.fusion_loss(fused_features, encoded_features)
        # loss_list2 = list()
        # loss_list2.append(lmd * consistency_loss + gamma * fusion_loss)
        # lsp_loss = mvc_loss.lsp_loss(fused_features)

        network_loss = sum(loss_list) + lmd * consistency_loss

        # 添加L1正则化
        l1_reg = 0.
        for param in network_model.parameters():
            l1_reg += param.abs().sum()
        network_loss += 0.00001 * l1_reg

        # 更新网络
        optimizer.zero_grad()
        network_loss.backward()
        optimizer.step()

        total_loss += network_loss.item()

    # if epoch % 10 == 0:
    print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss / num_samples))

    return total_loss


# def inference(network_model, mv_data, batch_size):
#     network_model.eval()  # 确保模型处于评估模式
#     mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
#
#     labels_vector = []  # 用于存储真实标签
#     unified_probs_list = []  # 用于收集所有批次的统一概率
#     TSNE_features_list = []
#
#     for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
#         with torch.no_grad():
#             lbps, _, features, _ = network_model(sub_data_views)
#             # 计算统一概率
#             batch_unified_probs = torch.mean(torch.stack(lbps), dim=0)
#             unified_probs_list.append(batch_unified_probs)
#             labels_vector.extend(sub_labels)
#
#             # fused_features = network_model.fuse_features(features)
#             TSNE_features_list.append(features.cpu())
#
#     # 将所有批次的统一概率合并，并计算最终预测标签
#     unified_probs = torch.cat(unified_probs_list, dim=0)
#     final_pred_labels = torch.argmax(unified_probs, dim=1).cpu().numpy()
#     TSNE_features = torch.cat(TSNE_features_list, dim=0).numpy()
#     labels_vector = np.array(labels_vector).reshape(-1)  # 确保标签向量是正确的形状
#
#     return final_pred_labels, labels_vector, TSNE_features


def inference(network_model, mv_data, batch_size, missing_info):
    network_model.eval()  # 确保模型处于评估模式
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    labels_vector = []  # 用于存储真实标签
    unified_probs_list = []  # 用于收集所有批次的统一概率
    TSNE_features_list = []

    for batch_idx, (sub_data_views, sub_labels, sub_missing_info) in enumerate(mv_data_loader):
        with torch.no_grad():
            lbps, _, features, _ = network_model(sub_data_views, sub_missing_info)
            # 初始化一个空的概率Tensor，用于累积非missing的概率
            cumulative_probs = torch.zeros_like(lbps[0])
            valid_counts = torch.zeros((cumulative_probs.size(0), 1), device=cumulative_probs.device)
            for idx, (lbp, missing) in enumerate(zip(lbps, sub_missing_info.t())):
                # 检查哪些样本在这个视图上不是missing
                valid = (missing == 0)
                cumulative_probs[valid] += lbp[valid]
                valid_counts[valid] += 1
            # 使用非missing视图的平均概率来填充
            unified_probs = cumulative_probs / valid_counts.clamp(min=1)  # 防止除以0
            unified_probs_list.append(unified_probs)
            labels_vector.extend(sub_labels)
            TSNE_features_list.append(features.cpu())

            # 将所有批次的统一概率合并，并计算最终预测标签
    unified_probs = torch.cat(unified_probs_list, dim=0)
    final_pred_labels = torch.argmax(unified_probs, dim=1).cpu().numpy()
    TSNE_features = torch.cat(TSNE_features_list, dim=0).numpy()  # 根据需要调整
    labels_vector = np.array(labels_vector).reshape(-1)  # 确保标签向量是正确的形状

    # 根据需要返回TSNE_features
    return final_pred_labels, labels_vector, TSNE_features


# def gaussian_kernel_similarity(x, y, sigma=1.0):
#     """计算两个向量之间的高斯核相似度"""
#     return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
#
#
# def build_similarity_matrix(features, sigma=1.0):
#     """根据特征构建高斯核相似度矩阵"""
#     n_samples = features.shape[0]
#     similarity_matrix = np.zeros((n_samples, n_samples))
#     for i in range(n_samples):
#         for j in range(i, n_samples):
#             similarity = gaussian_kernel_similarity(features[i], features[j], sigma=sigma)
#             similarity_matrix[i, j] = similarity
#             similarity_matrix[j, i] = similarity  # 相似度矩阵是对称的
#     return similarity_matrix
#
#
# def match_labels(true_labels, pred_labels):
#     """使用匈牙利算法匹配预测标签和真实标签"""
#     C = confusion_matrix(true_labels, pred_labels)
#     _, col_ind = linear_sum_assignment(-C)
#     matched_labels = np.zeros_like(pred_labels)
#     for i, p in enumerate(col_ind):
#         matched_labels[pred_labels == p] = i
#     return matched_labels


# def inference(network_model, mv_data, batch_size):
#     network_model.eval()  # 确保模型处于评估模式
#     mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
#
#     labels_vector = []  # 用于存储真实标签
#     fused_features_list = []
#
#     with torch.no_grad():
#         for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
#             lbps, _, features = network_model(sub_data_views)
#
#             fused_features = network_model.fuse_features(features)
#             fused_features_list.append(fused_features.cpu())
#
#             # batch_unified_probs = torch.mean(torch.stack(lbps), dim=0)
#             # batch_pred_labels = torch.argmax(batch_unified_probs, dim=1)
#             labels_vector.extend(sub_labels)
#
#     fused_features = torch.cat(fused_features_list, dim=0).numpy()
#     tsne = TSNE(n_components=2, random_state=42)
#     features_2d = tsne.fit_transform(fused_features)
#     # 使用高斯核构建相似度矩阵
#     # similarity_matrix = build_similarity_matrix(features_2d, sigma=1.0)
#     kmeans = KMeans(n_clusters=7, random_state=0)
#     kmeans.fit(features_2d)
#     # 应用谱聚类
#     # clustering = SpectralClustering(n_clusters=7, assign_labels='discretize')
#     # final_pred_labels = clustering.fit_predict(similarity_matrix)
#
#     labels_vector = np.array(labels_vector).reshape(-1)
#     # pred_labels = clustering.fit_predict(features_2d)
#     pred_labels = kmeans.labels_
#     # print(labels_vector[:20])
#     # 使用匈牙利算法匹配预测标签和真实标签
#     final_pred_labels = match_labels(labels_vector, pred_labels)
#     # print(final_pred_labels)
#
#     return final_pred_labels, labels_vector, fused_features


# def valid(network_model, mv_data, batch_size):
#
#     total_pred, pred_vectors, labels_vector = inference(network_model, mv_data, batch_size)
#     num_views = len(mv_data.data_views)
#
#     print("Clustering results on cluster assignments of each view:")
#     for idx in range(num_views):
#         acc, nmi, pur, ari = calculate_metrics(labels_vector,  pred_vectors[idx])
#         print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx+1, acc,
#                                                                                  idx+1, nmi,
#                                                                                  idx+1, pur,
#                                                                                  idx+1, ari))
#
#     print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
#     acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
#     print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))
#
#     return acc, nmi, pur, ari
def valid(network_model, mv_data, batch_size):
    total_pred, labels_vector, _ = inference(network_model, mv_data, batch_size)

    # 打印关于整体聚类结果的指标
    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari

# def extract_features_and_predict_labels(model, mv_data, batch_size):
#     mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
#     model.eval()
#
#     fused_features_list = []
#     predicted_labels_list = []  # Collect predicted labels here
#
#     with torch.no_grad():
#         for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
#             lbps, _, features = model(sub_data_views)
#
#             fused_features = model.fuse_features(features)
#             fused_features_list.append(fused_features.cpu())
#
#             batch_unified_probs = torch.mean(torch.stack(lbps), dim=0)
#             batch_pred_labels = torch.argmax(batch_unified_probs, dim=1)
#             predicted_labels_list.extend(batch_pred_labels.cpu().numpy())
#
#     fused_features = torch.cat(fused_features_list, dim=0).numpy()
#     predicted_labels = np.array(predicted_labels_list)
#
#     return fused_features, predicted_labels
