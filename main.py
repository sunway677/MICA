"""
Multi-View Contrastive Learning Framework for Clustering
This script implements a contrastive learning approach for multi-view clustering tasks.
It supports various datasets and includes functionality for model training, evaluation, and visualization.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import warnings
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set KMP duplicate lib environment variable to avoid warnings
warnings.filterwarnings("ignore")

# Import custom modules
from train import *
from layers import *
from loss import *

# Command line argument parser setup
parser = argparse.ArgumentParser(description='Deep Incomplete Multi-View Clustering via Multi-Level Imputation and Contrastive Alignment')
parser.add_argument('--load_pre_model', default=False, help='Load pre-trained model if True')
parser.add_argument('--load_full_model', default=False, help='Load fully trained model if True')
parser.add_argument('--save_model', default=True, help='Save the model after training')

# Dataset and training parameters
parser.add_argument('--db', type=str, default='MSRCv1',
                    choices=['MSRCv1', 'MNIST-USPS', 'scene', 'hand', 'Fashion', 'BDGP'],
                    help='Dataset name to use')
parser.add_argument('--seed', type=int, default=10, help='Random seed for reproducibility')
parser.add_argument("--mse_epochs", default=200, help='Number of epochs for pre-training')
parser.add_argument("--con_epochs", default=200, help='Number of epochs for fine-tuning')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay regularization parameter')
parser.add_argument("--temperature_l", type=float, default=1.0, help='Temperature parameter for contrastive loss')
parser.add_argument('--batch_size', default=100, type=int,
                    help='Batch size for training (total samples must be evenly divisible by batch_size)')
parser.add_argument('--normalized', type=bool, default=False, help='Whether to normalize features')
parser.add_argument('--gpu', default='0', type=str, help='GPU device index to use')

args = parser.parse_args()
print("==========\nArgs:{}\n==========".format(args))

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    """
    Set random seeds for reproducibility across all random number generators

    Args:
        seed (int): The random seed to use
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Dataset-specific parameter configurations
    if args.db == "MSRCv1":
        args.learning_rate = 0.0005
        args.batch_size = 35
        args.con_epochs = 200  # Reduced from 200
        args.seed = 10
        args.normalized = True

        # Network architecture dimensions
        dim_high_feature = 2000
        dim_low_feature = 1024
        dims = [256, 512]

        # Loss function hyperparameters
        alpha = 0.01  # Weight for view-specific contrastive loss
        beta = 0.005  # Weight for cross-view contrastive loss
        lmd = 0.01  # Weight for clustering loss
        gamma = 0.01  # Weight for regularization
        omega = 0.001  # Weight for fusion loss

    elif args.db == "MNIST-USPS":
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

    # Uncomment for hyperparameter grid search
    # for alpha in [0.007]:
    #     for beta in [0.01]:
    #         for args.learning_rate in [0.001]:

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Configure device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load multi-view data
    mv_data = MultiviewData(args.db, device)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = np.unique(mv_data.labels).size

    # Get input dimensions for each view
    input_sizes = np.zeros(num_views, dtype=int)
    for idx in range(num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

    # Start timing the training process
    t = time.time()

    # Initialize main network architecture
    mnw = MainNetwork(num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters,
                      args.batch_size)
    # Move network to GPU if available
    mnw = mnw.to(device)

    # Initialize loss function and optimizer
    mvc_loss = DeepMVCLoss(args.batch_size, num_clusters)
    optimizer = torch.optim.Adam(mnw.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.con_epochs, eta_min=0.00001)

    # Uncomment for separate optimizer for MLP fusion layer
    # optimizer_mlp = torch.optim.Adam(mnw.mlp_fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Load fully trained model for inference
    if args.load_full_model:
        state_dict = torch.load('./models/pytorch_full_model_%s.pth' % args.db)
        mnw.load_state_dict(state_dict)
        print("Loading full-trained model...")
        print("Starting inference...")

    # Load pre-trained model and continue training
    if args.load_pre_model:
        state_dict = torch.load('./models/pytorch_pre_model_%s.pth' % args.db)
        mnw.load_state_dict(state_dict)
        print("Loading pre-trained model...")
        print("Starting training and imputation...")

        # Fine-tuning with contrastive learning
        t = time.time()
        fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)
        for epoch in range(args.con_epochs):
            total_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, alpha, beta, lmd, gamma,
                                           omega, args.temperature_l, args.normalized, epoch, optimizer)
            fine_tuning_loss_values[epoch] = total_loss
            scheduler.step()
        print("Contrastive training completed.")
        print("Total time elapsed: {:.2f}s".format(time.time() - t))

    # Save the fully trained model
    if args.save_model:
        torch.save(mnw.state_dict(), './models/pytorch_full_model_%s.pth' % args.db)

    # Train from scratch if no pre-trained model is loaded
    if not args.load_full_model and not args.load_pre_model:
        # Pre-training phase
        pre_train_loss_values = pre_train(mnw, mv_data, args.batch_size, args.mse_epochs, optimizer)

        # Save pre-trained model
        if args.save_model:
            torch.save(mnw.state_dict(), './models/pytorch_pre_model_%s.pth' % args.db)

        # Fine-tuning phase with contrastive learning
        t = time.time()
        fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)
        for epoch in range(args.con_epochs):
            total_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, alpha, beta, lmd, gamma,
                                           omega, args.temperature_l, args.normalized, epoch, optimizer)
            fine_tuning_loss_values[epoch] = total_loss
            scheduler.step()
        print("Contrastive training completed.")
        print("Total time elapsed: {:.2f}s".format(time.time() - t))

    # Save the fully trained model
    if args.save_model:
        torch.save(mnw.state_dict(), './models/pytorch_full_model_%s.pth' % args.db)

    # Uncomment to perform inference and get predicted labels
    # predicted_labels, final_fused_labels = inference(mnw, mv_data, args.batch_size)

    # Uncomment for t-SNE visualization
    # predicted_labels, _, TSNE_features = inference(mnw, mv_data, args.batch_size)
    # tsne = TSNE(n_components=2, random_state=42)
    # features_2d = tsne.fit_transform(TSNE_features)
    #
    # plt.figure(figsize=(6, 5))
    # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=predicted_labels, cmap='viridis', alpha=0.6, s=1)
    # plt.colorbar(scatter, label='Cluster Labels')
    # plt.title('100 Epochs', fontsize=14)
    # plt.xticks([])  # Hide x-axis ticks
    # plt.yticks([])  # Hide y-axis ticks
    # # plt.xlabel('t-SNE 1')
    # # plt.ylabel('t-SNE 2')
    # plt.savefig('TSNEBDGP_100_Epochs.eps', format='eps')
    # plt.show()

    # Evaluate clustering performance
    acc, nmi, pur, ari = valid(mnw, mv_data, args.batch_size)

    # Save results to file
    with open('result_%s.txt' % args.db, 'a+') as f:
        f.write('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.4f} \n'.format(
            dim_high_feature, dim_low_feature, args.seed, args.batch_size,
            args.learning_rate, alpha, beta, acc, nmi, pur, (time.time() - t)))
        f.flush()