import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


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