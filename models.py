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
            phase_code = False  # no imputation
            # _, dvs, _, _ = network_model(sub_data_views, missing_info, phase_code)
            _, dvs, _, _, _, _ = network_model(sub_data_views, missing_info, phase_code)
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
                      normalized, epoch, optimizer):
    torch.autograd.set_detect_anomaly(True)
    network_model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    # criterion = torch.nn.MSELoss()
    total_loss = 0.
    all_encoded_features = []
    for batch_idx, (sub_data_views, _, missing_info) in enumerate(mv_data_loader):
        phase_code = True  # for imputation
        # lbps, dvs, _ , imputed_features = network_model(sub_data_views, missing_info, phase_code) # newly add missing_info
        lbps, dvs, _, imputed_features, encode_loss, decode_loss = network_model(sub_data_views, missing_info,
                                                                                 phase_code)
        # imputation_loss = encode_loss + decode_loss

        imputation_loss = encode_loss

        total_loss += imputation_loss.item()
        all_encoded_features.append(imputed_features)
        loss_list = list()
        for i in range(num_views):
            # missing_info中，1表示missing,0表示不missing的
            #mask = (1 - missing_info[:, i].unsqueeze(1)).expand_as(sub_data_views[i])
            for j in range(i + 1, num_views):
                loss_list.append(alpha * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))
                loss_list.append(beta * mvc_loss.forward_prob(lbps[i]))

        consistency_loss = mvc_loss.cross_view_consistency_loss(imputed_features)

        network_loss = sum(loss_list) + lmd * (consistency_loss+imputation_loss)

        #network_loss = sum(loss_list) + lmd * consistency_loss

        # 添加L1正则化
        l1_reg = 0.
        for param in network_model.parameters():
            l1_reg = l1_reg + param.abs().sum()
        network_loss = network_loss + 0.00001 * l1_reg

        # 更新网络
        optimizer.zero_grad()
        network_loss.backward()
        optimizer.step()

        total_loss += network_loss.item()

    # if epoch % 10 == 0:
    print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss / num_samples))

    return total_loss


def inference(network_model, mv_data, batch_size):
    network_model.eval()  # 确保模型处于评估模式
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    labels_vector = []  # 用于存储真实标签
    unified_probs_list = []  # 用于收集所有批次的统一概率
    TSNE_features_list = []
    phase_code = True

    for batch_idx, (sub_data_views, sub_labels, sub_missing_info) in enumerate(mv_data_loader):
        with torch.no_grad():
            # lbps, _, fused_prob, imputed_features = network_model(data_views, missing_info, phase_code)
            lbps, _, fused_features, features, _, _ = network_model(sub_data_views, sub_missing_info, phase_code)
            # lbps, _, features, _ = network_model(sub_data_views, sub_missing_info)

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
            TSNE_features_list.append(fused_features.cpu())

            # 将所有批次的统一概率合并，并计算最终预测标签
    unified_probs = torch.cat(unified_probs_list, dim=0)
    final_pred_labels = torch.argmax(unified_probs, dim=1).cpu().numpy()
    TSNE_features = torch.cat(TSNE_features_list, dim=0).numpy()
    labels_vector = np.array(labels_vector).reshape(-1)

    # 根据需要返回TSNE_features
    return final_pred_labels, labels_vector, TSNE_features


def valid(network_model, mv_data, batch_size):
    total_pred, labels_vector, _ = inference(network_model, mv_data, batch_size)

    # 打印关于整体聚类结果的指标
    print("Clustering results: ")
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari
