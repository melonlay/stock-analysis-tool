"""
DNN 模型定義
"""
import os
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    殘差塊
    """

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)

        # 如果輸入和輸出維度不同，添加投影層
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out


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
        activation: str = 'relu',
        l1_lambda: float = 0.01,
        gradient_clip: float = 1.0
    ):
        """
        初始化 DNN 模型

        Args:
            input_size (int): 輸入特徵維度
            hidden_sizes (List[int]): 隱藏層大小列表
            dropout_rate (float, optional): Dropout 比率。預設為 0.2
            batch_norm (bool, optional): 是否使用批次正規化。預設為 True
            activation (str, optional): 激活函數名稱。預設為 'relu'
            l1_lambda (float, optional): L1 正規化係數。預設為 0.01
            gradient_clip (float, optional): 梯度裁剪閾值。預設為 1.0
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation = activation
        self.l1_lambda = l1_lambda
        self.gradient_clip = gradient_clip

        # 構建網路層
        layers = []
        prev_size = input_size

        # 添加殘差塊
        for hidden_size in hidden_sizes:
            layers.append(ResidualBlock(prev_size, hidden_size, dropout_rate))
            prev_size = hidden_size

        # 輸出層
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

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
                # 使用 Kaiming 初始化
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x (torch.Tensor): 輸入張量

        Returns:
            torch.Tensor: 輸出張量
        """
        return self.model(x)

    def get_l1_loss(self) -> torch.Tensor:
        """
        計算 L1 正規化損失

        Returns:
            torch.Tensor: L1 正規化損失
        """
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return self.l1_lambda * l1_loss

    def clip_gradients(self):
        """
        裁剪梯度
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

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
            f"L1正規化係數: {self.l1_lambda}",
            f"梯度裁剪閾值: {self.gradient_clip}",
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
                'activation': self.activation,
                'l1_lambda': self.l1_lambda,
                'gradient_clip': self.gradient_clip
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
            activation=config['activation'],
            l1_lambda=config.get('l1_lambda', 0.01),
            gradient_clip=config.get('gradient_clip', 1.0)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_dnn_model(
    input_size: int,
    hidden_sizes: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    batch_norm: bool = True,
    activation: str = 'relu',
    l1_lambda: float = 0.01,
    gradient_clip: float = 1.0
) -> DNN:
    """
    創建 DNN 模型

    Args:
        input_size (int): 輸入特徵維度
        hidden_sizes (Optional[List[int]], optional): 隱藏層大小列表。預設為 None
        dropout_rate (float, optional): Dropout 比率。預設為 0.2
        batch_norm (bool, optional): 是否使用批次正規化。預設為 True
        activation (str, optional): 激活函數名稱。預設為 'relu'
        l1_lambda (float, optional): L1 正規化係數。預設為 0.01
        gradient_clip (float, optional): 梯度裁剪閾值。預設為 1.0

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
        activation=activation,
        l1_lambda=l1_lambda,
        gradient_clip=gradient_clip
    )
