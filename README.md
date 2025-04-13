MICA: Deep Incomplete Multi-View Clustering via Multi-Level Imputation and Contrastive Alignment
Overview
This repository implements MICA (Deep Incomplete Multi-View Clustering via Multi-Level Imputation and Contrastive Alignment), a novel approach for clustering with incomplete multi-view data. The framework combines sophisticated multi-level imputation mechanisms with contrastive learning to effectively handle missing data while discovering meaningful clusters across views.
Features

Multi-view support: Works with various types of multi-view datasets
Missing data handling: Robust mechanism for imputing missing features
Contrastive learning: Leverages contrastive loss for better feature representation
View-agnostic clustering: Unified clustering results from all available views
Comprehensive evaluation: Multiple metrics to evaluate clustering performance

Requirements

Python 3.6+
PyTorch 1.7+
NumPy
scikit-learn
SciPy

Key Parameters

--db: Dataset name (choices: MSRCv1, MNIST-USPS, scene, hand, Fashion, BDGP)
--seed: Random seed for reproducibility
--mse_epochs: Number of epochs for pre-training
--con_epochs: Number of epochs for fine-tuning
--learning_rate: Initial learning rate
--batch_size: Batch size for training
--gpu: GPU device ID to use

Pre-trained Models
You can use pre-trained models by setting:
bash--load_pre_model True
Or fully trained models:
bash--load_full_model True
Model Architecture
The framework consists of several key components:
![Framework](https://github.com/user-attachments/assets/261b5dd8-e837-42dc-b125-11bbc0fdb83a)

Encoders/Decoders: View-specific neural networks for feature extraction
Missing Data Imputation: KNN-based mechanism for handling missing features
Contrastive Loss: Both cluster-level and instance-level contrastive learning
Fusion Module: Combines information from all available views

Citation
If you find this code useful for your research, please cite our paper:
@article{wang2025deep,
  title={Deep Incomplete Multi-view Clustering via Multi-level Imputation and Contrastive Alignment},
  author={Wang, Ziyu and Du, Yiming and Wang, Yao and Ning, Rui and Li, Lusi},
  journal={Neural Networks},
  volume={181},
  pages={106851},
  year={2025},
  publisher={Elsevier}
}





