import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims, dropout_p=0.5):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims) + 1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i, nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims):  # No batch norm or dropout after last layer
                self.encoder.add_module('BatchNorm%d' % i, nn.BatchNorm1d(dims[i]))
                self.encoder.add_module('Dropout%d' % i, nn.Dropout(dropout_p))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims, dropout_p=0.5):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))  # Reverse dims to construct the decoder
        for i in range(len(dims) + 1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i, nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims):  # Avoid applying batch norm or dropout to the output layer
                self.decoder.add_module('BatchNorm%d' % i, nn.BatchNorm1d(dims[i]))
                self.decoder.add_module('Dropout%d' % i, nn.Dropout(dropout_p))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)



# cross-view feature融合
class CVCLNetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
        super(CVCLNetwork, self).__init__()

        self.encoders = list()
        self.decoders = list()
        self.encode_features = [nn.Parameter(torch.zeros(35, dim_high_feature)) for _ in range(num_views)]

        for idx in range(num_views):
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

    # 计算特征的平均值作为融合策略
    # def fuse_features(self, encoded_features):
    #     fused_features = torch.mean(torch.stack(encoded_features), dim=0)
    #     return fused_features


    def forward(self, data_views, missing_info):
        global high_features, label_probs
        lbps = list()
        dvs = list()
        features = list()

        num_views = len(data_views)
        # print(data_views[0].shape)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # 获取所有视图的encoding特征
        # encoded_features = [self.encoders[idx](data_views[idx].to(device).float()) for idx in range(num_views)]

        # 使用融合方法融合特征
        # fused_features = self.fuse_features(encoded_features)

        # 计算每个视图的预测概率并存储
        # for encoded_feature in encoded_features:
        #     label_probs = self.label_learning_module(encoded_feature)
        #     lbps.append(label_probs)

        # 根据预测概率计算权重（使用最大预测概率的平均值）
        # weights = [torch.max(lbp, dim=1)[0].mean() for lbp in lbps]
        # total_weight = sum(weights)
        # normalized_weights = [weight / total_weight for weight in weights]
        # # 使用计算得到的权重对特征进行加权融合
        # fused_features = torch.zeros_like(encoded_features[0])
        # for weight, feature in zip(normalized_weights, encoded_features):
        #     fused_features += weight * feature

        for idx in range(num_views):
            valid_idx = (missing_info[:, idx] == 0)
            print(valid_idx)
            data_view = data_views[idx].to(device).float()
            high_features = self.encoders[idx](data_view)

            if valid_idx.any():
                # 选择不missing的特征进行标签预测概率计算
                valid_encoded_feature = high_features[valid_idx]
                label_probs = self.label_learning_module(valid_encoded_feature)
                # 为了保持lbps列表的维度一致性，创建一个与完整batch大小相同的填充Tensor
                padded_probs = torch.zeros(data_view.size(0), label_probs.size(1), device=device)
                padded_probs[valid_idx] = label_probs
                lbps.append(padded_probs)


            # label_probs = self.label_learning_module(high_features)
            # print(high_features.shape)

            # unified_probs = self.label_learning_module(high_features)
            # 使用融合后的特征进行decode
            data_view_recon = self.decoders[idx](high_features)
            dvs.append(data_view_recon)

            features.append(high_features)
            lbps.append(label_probs)

        return lbps, dvs, high_features, features
