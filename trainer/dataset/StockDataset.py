"""
股票資料集類別
"""
from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


class StockDataset(Dataset):
    """
    股票資料集類別
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        初始化資料集

        Args:
            features (np.ndarray): 特徵矩陣
            labels (np.ndarray): 標籤陣列
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
