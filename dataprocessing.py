"""
Multi-View Dataset Processing Module
This module implements dataset loading and preprocessing functionality for multi-view clustering.
It supports various multi-view datasets with handling for missing data.
"""

import os
import random
import sys
import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class MultiviewData(Dataset):
    """
    Dataset class for multi-view data with support for missing data patterns
    Implements PyTorch Dataset interface for easy integration with DataLoader
    """

    def __init__(self, db, device, path="datasets/"):
        """
        Initialize the multi-view dataset

        Args:
            db (str): Name of the dataset to load
            device (torch.device): Device to load the tensors onto (CPU or GPU)
            path (str): Directory path where the datasets are stored
        """
        self.data_views = list()
        self.device = device  # Store the device for tensor allocation

        # Load and preprocess dataset based on the specified name
        if db == "MSRCv1":
            # Microsoft Research Cambridge v1 dataset (with 0.1 missing rate)
            mat = sio.loadmat(os.path.join(path, 'MSRCv1_missing_0.1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]

            # Extract views from cell array
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))

            # Normalize data using Min-Max scaling
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            # Extract labels and missing information
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif db == "MNIST-USPS":
            # Combined MNIST and USPS handwritten digits dataset (with 0.1 missing rate)
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS_missing_0.1.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)

            # Flatten 2D images to 1D vectors
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)

            # Extract labels and missing information
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif db == "BDGP":
            # Berkeley Drosophila Genome Project dataset (with 0.1 missing rate)
            mat = sio.loadmat(os.path.join(path, 'BDGP_missing_0.1.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)

            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)

            # Extract labels and missing information
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif db == "Fashion":
            # Fashion dataset (multi-view)
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))

            # Reshape 2D features to 1D vectors for each view
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)

            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)

            # Extract labels (no missing information for this dataset)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "hand":
            # Handwritten digits dataset (with 0.1 missing rate)
            mat = sio.loadmat(os.path.join(path, 'handwritten_missing_0.1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]

            # Extract views from cell array
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))

            # Normalize data using Min-Max scaling
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            # Extract labels and missing information (with +1 correction for labels)
            self.labels = np.array(np.squeeze(mat['Y']) + 1).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        elif db == "scene":
            # Scene15 dataset (with 0.1 missing rate)
            mat = sio.loadmat(os.path.join(path, 'Scene15_missing_0.1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]

            # Extract views from cell array
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))

            # Normalize data using Min-Max scaling
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])

            # Extract labels and missing information
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
            self.missing_matrix = torch.tensor(mat['missing_matrix'], dtype=torch.float32).to(self.device)

        else:
            raise NotImplementedError(f"Dataset {db} is not implemented")

        # Convert all data views to PyTorch tensors on the specified device
        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device).float()

        # Ensure labels are properly extracted
        self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns:
            int: Number of samples
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            tuple: (list of view data, label, missing info tensor)
        """
        sub_data_views = list()
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]
            sub_data_views.append(data_view[index])

        # Get missing information for this sample
        missing_info = self.missing_matrix[index, :]

        return sub_data_views, self.labels[index], missing_info


def get_multiview_data(mv_data, batch_size):
    """
    Create a DataLoader for multi-view data with the specified batch size

    Args:
        mv_data (MultiviewData): The multi-view dataset
        batch_size (int): Batch size for the DataLoader

    Returns:
        tuple: (DataLoader, number of views, number of samples, number of clusters)
    """
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    # Create DataLoader with shuffling and fixed batch size
    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Drop last batch if it's smaller than batch_size
    )

    return mv_data_loader, num_views, num_samples, num_clusters