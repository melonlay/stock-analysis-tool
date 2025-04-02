"""
深度神經網路模型定義
"""
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    """
    深度神經網路模型

    此模型包含多個全連接層，每層後面接續批次正規化、ReLU激活函數和Dropout。
    最後一層使用sigmoid激活函數，適合二分類問題。
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        activation: str = 'relu'
    ) -> None:
        """
        初始化DNN模型

        Args:
            input_size (int): 輸入特徵的維度
            hidden_sizes (List[int]): 隱藏層的神經元數量列表
            dropout_rate (float, optional): Dropout比率。預設為 0.2
            batch_norm (bool, optional): 是否使用批次正規化。預設為 True
            activation (str, optional): 激活函數類型。預設為 'relu'
        """
        super(DNN, self).__init__()

        # 保存模型參數
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation = activation

        # 建立層列表
        self.layers = nn.ModuleList()

        # 添加輸入層到第一個隱藏層
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))

        # 添加隱藏層
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))

        # 添加輸出層
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

        # 初始化權重
        self._init_weights()

    def _init_weights(self) -> None:
        """
        初始化模型權重
        使用 Kaiming 初始化方法，適合 ReLU 激活函數
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_activation(self) -> nn.Module:
        """
        獲取激活函數

        Returns:
            nn.Module: 激活函數層
        """
        if self.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.activation.lower() == 'leaky_relu':
            return nn.LeakyReLU()
        elif self.activation.lower() == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"不支援的激活函數類型: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, input_size)

        Returns:
            torch.Tensor: 輸出張量，形狀為 (batch_size, 1)
        """
        activation = self._get_activation()

        # 前向傳播
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
                x = activation(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 最後一層使用 sigmoid 激活函數
        return torch.sigmoid(x)

    def get_model_summary(self) -> str:
        """
        獲取模型摘要

        Returns:
            str: 包含模型架構和參數數量的摘要
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        summary = f"""
DNN 模型摘要:
輸入維度: {self.input_size}
隱藏層: {self.hidden_sizes}
Dropout率: {self.dropout_rate}
批次正規化: {'是' if self.batch_norm else '否'}
激活函數: {self.activation}
總參數數量: {total_params:,}
可訓練參數數量: {trainable_params:,}
"""
        return summary

    def save_model(self, path: str) -> None:
        """
        保存模型

        Args:
            path (str): 模型保存路徑
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'activation': self.activation
        }, path)

    @classmethod
    def load_model(cls, path: str) -> 'DNN':
        """
        載入模型

        Args:
            path (str): 模型檔案路徑

        Returns:
            DNN: 載入的模型實例
        """
        checkpoint = torch.load(path)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_sizes=checkpoint['hidden_sizes'],
            dropout_rate=checkpoint['dropout_rate'],
            batch_norm=checkpoint['batch_norm'],
            activation=checkpoint['activation']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_dnn_model(
    input_size: int,
    hidden_sizes: List[int],
    dropout_rate: float = 0.2,
    batch_norm: bool = True,
    activation: str = 'relu'
) -> DNN:
    """
    創建DNN模型的工廠函數

    Args:
        input_size (int): 輸入特徵的維度
        hidden_sizes (List[int]): 隱藏層的神經元數量列表
        dropout_rate (float, optional): Dropout比率。預設為 0.2
        batch_norm (bool, optional): 是否使用批次正規化。預設為 True
        activation (str, optional): 激活函數類型。預設為 'relu'

    Returns:
        DNN: 創建的DNN模型實例
    """
    return DNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        activation=activation
    )
