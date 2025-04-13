"""
Data Processing for Multi-View Learning with Incomplete Data
This module handles the creation of incomplete multi-view data for experimental evaluation.
It ensures that each instance appears in at least one view while creating controlled missing patterns.
"""

import numpy as np
from scipy.io import loadmat, savemat
import os


def make_data_incomplete_and_track_missing(X, missing_ratio):
    """
    Process data from each view according to the specified missing ratio,
    ensuring each instance appears in at least one view.
    Also tracks and returns a matrix indicating which instances are missing in which views.

    Args:
        X: List of arrays or dictionary of arrays. Each element is a 2D array or a dictionary of 2D arrays.
        missing_ratio: The specified missing ratio, between 0 and 1.

    Returns:
        tuple: (Processed X with missing data, Boolean matrix indicating missing positions)
    """
    # Determine data structure and dimensions
    if isinstance(X, dict):
        # X is a dictionary of arrays
        num_views = len(X)
        view_keys = list(X.keys())
        num_instances = X[view_keys[0]].shape[0]
    else:
        # X is a list of arrays
        num_views = len(X)
        num_instances = X[0].shape[0]

    # Initialize tracking matrix for missing data (True = missing)
    missing_matrix = np.zeros((num_instances, num_views), dtype=bool)

    # Calculate per-view missing ratio
    ratio = missing_ratio / num_views

    # Randomly select one protected view for each instance to ensure it appears at least once
    protected_views_for_instances = np.random.randint(num_views, size=num_instances)

    # Process each view
    for view_index in range(num_views):
        # Get current view data
        if isinstance(X, dict):
            view_data = X[view_keys[view_index]]
        else:
            view_data = X[view_index]

        # Calculate number of instances to remove in this view
        num_missing_instances = int(np.floor(ratio * num_instances))

        # Find indices that can be removed (excluding protected instances)
        possible_indices = [i for i in range(num_instances) if protected_views_for_instances[i] != view_index]

        # Choose indices to remove
        if len(possible_indices) < num_missing_instances:
            indices_to_remove = possible_indices
        else:
            indices_to_remove = np.random.choice(possible_indices, num_missing_instances, replace=False)

        # Set selected instances as missing in current view
        for idx in indices_to_remove:
            view_data[idx] = 0  # Set to zero to indicate missing
            missing_matrix[idx, view_index] = True  # Track missing position

        # Update the data structure
        if isinstance(X, dict):
            X[view_keys[view_index]] = view_data
        else:
            X[view_index] = view_data

    return X, missing_matrix


def main():
    """
    Main function to process datasets and create incomplete multi-view data.
    Handles both Cell Array datasets and dictionary-structured datasets.
    """
    # Process Cell Array dataset (Scene15)
    mat_data = loadmat('./datasets/Scene15.mat', squeeze_me=True)
    X = mat_data['X']  # Cell Array data
    # Alternative data key: X = mat_data['data']
    Y = mat_data['Y']  # Labels
    # Alternative label key: Y = mat_data['truth']
    Y = Y.reshape(-1, 1)  # Ensure labels are in correct shape
    missing_ratio = 0.7  # Set missing ratio (70%)

    # Process data and track missing information
    X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)

    # Save processed Cell Array data and missing matrix
    original_filename = 'Scene15.mat'
    original_filename_without_extension = os.path.splitext(original_filename)[0]
    new_filename = f"{original_filename_without_extension}_missing_{missing_ratio}.mat"

    # Update the data structure
    mat_data['X'] = X_incomplete
    mat_data['Y'] = Y
    mat_data['missing_matrix'] = missing_matrix  # Save missing indicator matrix
    savemat(new_filename, mat_data)
    print(f"Processed data and missing matrix saved as: {new_filename}")

    # Code for processing non-Cell Array dataset (commented out)
    """
    # Process non-Cell Array dataset (Fashion)
    mat_data = loadmat('./datasets/Fashion.mat', squeeze_me=True)
    X = {key: mat_data[key] for key in mat_data.keys() if key.startswith('X')}  # Dictionary of views
    Y = mat_data['Y']  # Labels
    Y = Y.reshape(-1, 1)  # Ensure labels are in correct shape
    missing_ratio = 0.7  # Set missing ratio (70%)

    # Process data and track missing information
    X_incomplete, missing_matrix = make_data_incomplete_and_track_missing(X, missing_ratio)

    # Save processed non-Cell Array data and missing matrix
    original_filename = 'Fashion.mat'
    original_filename_without_extension = os.path.splitext(original_filename)[0]
    new_filename = f"{original_filename_without_extension}_missing_{missing_ratio}.mat"

    # Update the data structure
    mat_data.update(X_incomplete)
    mat_data['Y'] = Y
    mat_data['missing_matrix'] = missing_matrix  # Save missing indicator matrix
    savemat(new_filename, mat_data)
    print(f"Processed data and missing matrix saved as: {new_filename}")
    """


if __name__ == "__main__":
    main()