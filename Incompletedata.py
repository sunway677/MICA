import numpy as np
from scipy.io import loadmat, savemat
import os


def make_data_incomplete_and_track_missing(X, missing_ratio):
    """
    对于每个view的数据根据指定的missing ratio来处理数据，并确保每个instance至少在一个view中出现一次。
    同时，记录并返回一个矩阵，指示哪些instance在哪些view中被missing了。

    :param X: cell array, 每个cell是一个二维数组。
    :param missing_ratio: 指定的missing ratio，范围在0到1之间。
    :return: 处理后的X和一个布尔矩阵，指示missing的位置。
    """
    num_views = len(X)
    print(num_views)
    num_instances = X[0].shape[0]
    print(num_instances)
    # print(X[0][0].shape)

    missing_matrix = np.zeros((num_instances, num_views), dtype=bool)

    # 为每个instance随机选择一个保护view
    protected_views_for_instances = np.random.randint(num_views, size=num_instances)

    for view_index in range(num_views):
        view_data = X[view_index]
        num_missing_instances = int(np.floor(missing_ratio * num_instances))
        possible_indices = [i for i in range(num_instances) if protected_views_for_instances[i] != view_index]

        if len(possible_indices) < num_missing_instances:
            indices_to_remove = possible_indices
        else:
            indices_to_remove = np.random.choice(possible_indices, num_missing_instances, replace=False)

        # 将选中的instances在当前view中设置为missing
        for idx in indices_to_remove:
            view_data[idx] = 0
            missing_matrix[idx, view_index] = True

    return X, missing_matrix


def main():
    mat_data = loadmat('MSRCv1.mat', squeeze_me=True)
    X = mat_data['X']
    Y = mat_data['Y']
    Y = Y.reshape(-1, 1)
    missing_ratio = 0.7

    # 对数据进行missing处理并记录missing信息
    X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)

    # 保存处理后的数据及missing矩阵
    original_filename = 'MSRCv1.mat'
    original_filename_without_extension = os.path.splitext(original_filename)[0]
    new_filename = f"{original_filename_without_extension}_missing_{missing_ratio}.mat"
    mat_data['X'] = X_incomplete
    mat_data['Y'] = Y
    mat_data['missing_matrix'] = missing_matrix  # 保存missing矩阵
    savemat(new_filename, mat_data)
    print(f"处理后的数据及missing矩阵已保存为：{new_filename}")


if __name__ == "__main__":
    main()
