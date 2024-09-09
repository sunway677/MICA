import argparse
import warnings

from models import *
from layers import *
from loss import *

import torch
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MNIST-USPS')
parser.add_argument('--load_pre_model', default=True, help='Testing if True or pre-training.')
parser.add_argument('--load_full_model', default=False, help='Testing if True or training.')
parser.add_argument('--save_model', default=True, help='Saving the model after training.')

parser.add_argument('--db', type=str, default='MSRCv1',
                    choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP',
                             '100leaves'],
                    help='dataset name')
parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
parser.add_argument("--mse_epochs", default=100, help='Number of epochs to pre-training.')  # 200
parser.add_argument("--con_epochs", default=200, help='Number of epochs to fine-tuning.')  # 100
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initializing learning rate.')
parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument('--batch_size', default=100, type=int,
                    help='The total number of samples must be evenly divisible by batch_size.')
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')

args = parser.parse_args()
print("==========\nArgs:{}\n==========".format(args))

# torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    if args.db == "MSRCv1":
        # db checked 97.62
        args.learning_rate = 0.001  #Yiming
        args.batch_size = 50
        args.con_epochs = 50  # 200
        args.seed = 10
        args.normalized = True

        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]
        alpha = 0.07  #Yiming
        beta = 0.01  # Yiming
        lmd = 0.01
        gamma = 0.0
        omega = 0.001

    elif args.db == "MNIST-USPS":
        # db checked 99.7
        args.learning_rate = 0.0001
        args.batch_size = 50
        args.seed = 10
        args.con_epochs = 100
        args.normalized = False

        dim_high_feature = 1500
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        alpha = 0.03
        beta = 0.07
        lmd = 0.01
        gamma = 0.01
        omega = 0.001

    elif args.db == "COIL20":
        # db checked 84.65
        args.learning_rate = 0.0005
        args.batch_size = 180
        args.seed = 50
        args.con_epochs = 400
        args.normalized = False

        dim_high_feature = 768
        dim_low_feature = 200
        dims = [256, 512, 1024, 2048]
        alpha = 0.01
        beta = 0.01
        lmd = 0.01
        gamma = 0.01
        omega = 0.001

    elif args.db == "scene":
        # db checked 44.59
        args.learning_rate = 0.0005
        args.con_epochs = 200
        args.batch_size = 69
        args.seed = 10
        args.normalized = False

        dim_high_feature = 1500
        dim_low_feature = 256
        dims = [256, 512, 1024, 2048]
        alpha = 0.01
        beta = 0.05
        lmd = 0.01
        gamma = 0.01
        omega = 0.001

    elif args.db == "hand":
        # db checked 96.85
        args.learning_rate = 0.0001
        args.batch_size = 200
        args.seed = 50
        args.con_epochs = 100
        args.normalized = True

        dim_high_feature = 1024
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        alpha = 0.005
        beta = 0.001
        lmd = 0.01
        gamma = 0.001
        omega = 0.001

    elif args.db == "Fashion":
        # db checked 99.31
        args.learning_rate = 0.0005
        args.batch_size = 100
        args.con_epochs = 100
        args.seed = 20
        args.normalized = True
        args.temperature_l = 0.5

        dim_high_feature = 2000
        dim_low_feature = 500
        dims = [256, 512]
        alpha = 0.005
        beta = 0.005
        lmd = 0.01
        gamma = 0.01
        omega = 0.001

    elif args.db == "BDGP":
        # db checked 99.2
        args.learning_rate = 0.0001  #0.0001
        args.batch_size = 250
        args.seed = 10
        args.con_epochs = 100
        args.normalized = True

        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]
        alpha = 0.06  #0.06
        beta = 0.01
        lmd = 0.01
        gamma = 0.01
        omega = 0.001

    elif args.db == "mfeat":
        # db checked 99.2
        args.learning_rate = 0.0001
        args.batch_size = 50
        args.seed = 10
        args.con_epochs = 100
        args.normalized = True

        dim_high_feature = 2048
        dim_low_feature = 512
        dims = [256, 512, 1024, 2048]
        alpha = 0.05
        beta = 0.05
        lmd = 0.001
        gamma = 0.001
        omega = 0.01

    # for alpha in [0.007]:
    #     for beta in [0.01]:
    #         for args.learning_rate in [0.001]:

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mv_data = MultiviewData(args.db, device)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = np.unique(mv_data.labels).size

    input_sizes = np.zeros(num_views, dtype=int)
    for idx in range(num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

    t = time.time()
    # neural network architecture
    mnw = CVCLNetwork(num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters,
                      args.batch_size)
    # filling it into GPU
    mnw = mnw.to(device)

    mvc_loss = DeepMVCLoss(args.batch_size, num_clusters)
    optimizer = torch.optim.Adam(mnw.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.con_epochs, eta_min=0.00001)
    # optimizer_mlp = torch.optim.Adam(mnw.mlp_fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.load_full_model:  # loading fully trained model, and directlt perform inference
        state_dict = torch.load('./models/CVCL_pytorch_full_model_%s.pth' % args.db)
        mnw.load_state_dict(state_dict)
        print("Loading full-trained model...")
        print("Staring inference...")

    if args.load_pre_model:  # only loading pre-training model (pretrained for 200 epoches), and then training
        state_dict = torch.load('./models/CVCL_pytorch_pre_model_%s.pth' % args.db)
        mnw.load_state_dict(state_dict)
        print("Loading pre-trained model...")
        print("Staring training and imputation...")
        ###only training
        t = time.time()
        fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)
        for epoch in range(args.con_epochs):
            total_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, alpha, beta, lmd, gamma,
                                           omega,
                                           args.temperature_l, args.normalized, epoch,
                                           optimizer)
            fine_tuning_loss_values[epoch] = total_loss
            scheduler.step()
        print("contrastive_train finished.")
        print("Total time elapsed: {:.2f}s".format(time.time() - t))

    ### Saving full model
    if args.save_model:
        torch.save(mnw.state_dict(), './models/CVCL_pytorch_full_model_%s.pth' % args.db)

    if not args.load_full_model and not args.load_pre_model:  # pre-training, training from scrach
        pre_train_loss_values = pre_train(mnw, mv_data, args.batch_size, args.mse_epochs, optimizer)

        if args.save_model:
            torch.save(mnw.state_dict(), './models/CVCL_pytorch_pre_model_%s.pth' % args.db)

        t = time.time()
        fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)
        for epoch in range(args.con_epochs):
            total_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, alpha, beta, lmd, gamma,
                                           omega,
                                           args.temperature_l, args.normalized, epoch,
                                           optimizer)
            fine_tuning_loss_values[epoch] = total_loss
            scheduler.step()
        print("contrastive_train finished.")
        print("Total time elapsed: {:.2f}s".format(time.time() - t))

    if args.save_model:
        torch.save(mnw.state_dict(), './models/CVCL_pytorch_full_model_%s.pth' % args.db)

    # predicted_labels, final_fused_labels = inference(mnw, mv_data, args.batch_size)

    # #TSNE 可视化
    # predicted_labels, _, TSNE_features = inference(mnw, mv_data, args.batch_size)
    # tsne = TSNE(n_components=2, random_state=42)
    # features_2d = tsne.fit_transform(TSNE_features)
    #
    # plt.figure(figsize=(6, 5))
    # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=predicted_labels, cmap='viridis', alpha=0.6,s=1)
    # plt.colorbar(scatter, label='Cluster Labels')
    # plt.title('100 Epochs',fontsize=14)
    # plt.xticks([])  # 隐藏 x 轴刻度
    # plt.yticks([])
    # # plt.xlabel('t-SNE 1')
    # # plt.ylabel('t-SNE 2')
    # plt.savefig('TSNEBDGP_100_Epochs.eps', format='eps')
    # plt.show()

    acc, nmi, pur, ari = valid(mnw, mv_data, args.batch_size)
    with open('result_%s.txt' % args.db, 'a+') as f:
        f.write('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.4f} \n'.format(
            dim_high_feature, dim_low_feature, args.seed, args.batch_size,
            args.learning_rate, alpha, beta, acc, nmi, pur, (time.time() - t)))
        f.flush()
