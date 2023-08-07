"""
nn_pipeline.py - Custom data pipeline for PyTorch DataLoader

This module defines a custom data pipeline class, DataPipeline, for loading and converting the data into PyTorch DataLoader objects. 
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd


DEVICE = 'cuda'

class DataPipeline(Dataset):
    """
    Custom data pipeline for preparing and loading data into PyTorch DataLoader.
    
    This class takes training and testing data along with relevant parameters and creates
    DataLoader objects for both training and testing datasets.
    
    Args:
        X_train: Training feature data.
        X_test: Testing feature data.
        y_train: Training target data.
        y_test: Testing target data.
        batch_size: Batch size for DataLoader.
        device: Device to which data should be moved. Default is cuda.
    """
    def __init__(
        self,
        X_train:pd.DataFrame, 
        X_test:pd.DataFrame, 
        y_train:pd.Series, 
        y_test:pd.Series,
        batch_size: int,
        device:str=DEVICE,
    ) -> None:

        # Number of features
        self.feature_num = X_train.shape[1]

        # Converting our data to tensors
        X_train = torch.tensor(X_train.values.astype('float32')).to(device)
        y_train = torch.tensor(y_train.values.astype('float32')).to(device)

        X_test = torch.tensor(X_test.values.astype('float32')).to(device)
        y_test = torch.tensor(y_test.values.astype('float32')).to(device)

        # Create Dataset objects
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # Storing the test and the train DataLoader
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=len(test_dataset),
            shuffle=False
        )
