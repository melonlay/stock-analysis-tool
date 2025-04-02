"""
DNN 模型定義
"""
import os
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn


class DNN(nn.Module):
    """
    深度神經網路模型
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        初始化 DNN 模型

        Args:
            input_size (int): 輸入特徵維度
            hidden_sizes (List[int]): 隱藏層大小列表
            dropout_rate (float, optional): Dropout 比率。預設為 0.2
            batch_norm (bool, optional): 是否使用批次正規化。預設為 True
            activation (str, optional): 激活函數名稱。預設為 'relu'
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation = activation

        # 構建網路層
        layers = []
        prev_size = input_size

        # 添加隱藏層
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(self._get_activation())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # 輸出層
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # 添加 Sigmoid 激活函數

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_activation(self) -> nn.Module:
        """
        獲取激活函數

        Returns:
            nn.Module: 激活函數模組
        """
        if self.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.activation.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif self.activation.lower() == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"不支援的激活函數: {self.activation}")

    def _initialize_weights(self):
        """
        初始化模型權重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用較小的初始化範圍
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x (torch.Tensor): 輸入張量

        Returns:
            torch.Tensor: 輸出張量
        """
        return self.model(x)

    def get_model_summary(self) -> str:
        """
        獲取模型摘要

        Returns:
            str: 模型摘要字符串
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        summary = [
            "\nDNN 模型摘要:",
            f"輸入維度: {self.input_size}",
            f"隱藏層: {self.hidden_sizes}",
            f"Dropout率: {self.dropout_rate}",
            f"批次正規化: {'是' if self.batch_norm else '否'}",
            f"激活函數: {self.activation}",
            f"總參數數量: {total_params:,}",
            f"可訓練參數數量: {trainable_params:,}\n"
        ]
        return "\n".join(summary)

    def save_model(self, path: str):
        """
        保存模型

        Args:
            path (str): 保存路徑
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'dropout_rate': self.dropout_rate,
                'batch_norm': self.batch_norm,
                'activation': self.activation
            }
        }, path)

    @classmethod
    def load_model(cls, path: str) -> 'DNN':
        """
        載入模型

        Args:
            path (str): 模型路徑

        Returns:
            DNN: 載入的模型
        """
        checkpoint = torch.load(path)
        config = checkpoint['config']

        model = cls(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate'],
            batch_norm=config['batch_norm'],
            activation=config['activation']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_dnn_model(
    input_size: int,
    hidden_sizes: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    batch_norm: bool = True,
    activation: str = 'relu'
) -> DNN:
    """
    創建 DNN 模型

    Args:
        input_size (int): 輸入特徵維度
        hidden_sizes (Optional[List[int]], optional): 隱藏層大小列表。預設為 None
        dropout_rate (float, optional): Dropout 比率。預設為 0.2
        batch_norm (bool, optional): 是否使用批次正規化。預設為 True
        activation (str, optional): 激活函數名稱。預設為 'relu'

    Returns:
        DNN: 創建的模型
    """
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]

    return DNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        activation=activation
    )
