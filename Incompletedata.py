import numpy as np
from scipy.io import loadmat, savemat
import os

def make_data_incomplete_and_track_missing(X, missing_ratio):
    """
    对于每个view的数据根据指定的missing ratio来处理数据，并确保每个instance至少在一个view中出现一次。
    同时，记录并返回一个矩阵，指示哪些instance在哪些view中被missing了。

    :param X: list of arrays or dict of arrays, 每个元素是一个二维数组或多个二维数组组成的字典。
    :param missing_ratio: 指定的missing ratio，范围在0到1之间。
    :return: 处理后的X和一个布尔矩阵，指示missing的位置。
    """
    if isinstance(X, dict):
        # X is a dictionary of arrays
        num_views = len(X)
        view_keys = list(X.keys())
        num_instances = X[view_keys[0]].shape[0]
    else:
        # X is a list of arrays
        num_views = len(X)
        num_instances = X[0].shape[0]

    missing_matrix = np.zeros((num_instances, num_views), dtype=bool)
    ratio = missing_ratio / num_views

    # 为每个instance随机选择一个保护view
    protected_views_for_instances = np.random.randint(num_views, size=num_instances)

    for view_index in range(num_views):
        if isinstance(X, dict):
            view_data = X[view_keys[view_index]]
        else:
            view_data = X[view_index]

        num_missing_instances = int(np.floor(ratio * num_instances))
        possible_indices = [i for i in range(num_instances) if protected_views_for_instances[i] != view_index]

        if len(possible_indices) < num_missing_instances:
            indices_to_remove = possible_indices
        else:
            indices_to_remove = np.random.choice(possible_indices, num_missing_instances, replace=False)

        # 将选中的instances在当前view中设置为missing
        for idx in indices_to_remove:
            view_data[idx] = 0
            missing_matrix[idx, view_index] = True

        if isinstance(X, dict):
            X[view_keys[view_index]] = view_data
        else:
            X[view_index] = view_data

    return X, missing_matrix

def main():
    # 处理Cell Array数据集
    mat_data = loadmat('./datasets/Scene15.mat', squeeze_me=True)
    X = mat_data['X']
    # X = mat_data['data']
    Y = mat_data['Y']
    # Y = mat_data['truth']
    Y = Y.reshape(-1, 1)
    missing_ratio = 0.7

    # 对MSRCv1数据进行missing处理并记录missing信息
    X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)

    # 保存处理后的Cell Array数据及missing矩阵
    original_filename = 'Scene15.mat'
    original_filename_without_extension = os.path.splitext(original_filename)[0]
    new_filename= f"{original_filename_without_extension}_missing_{missing_ratio}.mat"
    mat_data['X'] = X_incomplete
    mat_data['Y'] = Y
    mat_data['missing_matrix'] = missing_matrix  # 保存missing矩阵
    savemat(new_filename, mat_data)
    print(f"处理后的数据及missing矩阵已保存为：{new_filename}")


    # 处理非Cell Array数据集
    # mat_data = loadmat('./datasets/Fashion.mat', squeeze_me=True)
    # X = {key: mat_data[key] for key in mat_data.keys() if key.startswith('X')}
    # Y = mat_data['Y']
    # Y = Y.reshape(-1, 1)
    # missing_ratio = 0.7
    #
    # # 对非Cell Array数据进行missing处理并记录missing信息
    # X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)
    #
    # # 保存处理后的非Cell Array数据及missing矩阵
    # original_filename = 'Fashion.mat'
    # original_filename_without_extension = os.path.splitext(original_filename)[0]
    # new_filename = f"{original_filename_without_extension}_missing_{missing_ratio}.mat"
    # mat_data.update(X_incomplete)
    # mat_data['Y'] = Y
    # mat_data['missing_matrix'] = missing_matrix  # 保存missing矩阵
    # savemat(new_filename, mat_data)
    # print(f"处理后的数据及missing矩阵已保存为：{new_filename}")

if __name__ == "__main__":
    main()
