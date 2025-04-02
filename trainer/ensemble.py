"""
集成學習模組
"""
import logging
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class VotingEnsemble(nn.Module):
    """
    投票式集成模型

    將多個基礎模型的預測結果進行組合，採用加權投票的方式得出最終預測。
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        threshold: float = 0.5
    ):
        """
        初始化集成模型

        Args:
            models (List[nn.Module]): 基礎模型列表
            weights (Optional[List[float]], optional): 每個模型的權重。默認為None，表示等權重
            threshold (float, optional): 預測閾值。默認為0.5
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            total_weight = sum(weights)
            self.weights = torch.tensor([w / total_weight for w in weights])
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x (torch.Tensor): 輸入特徵

        Returns:
            torch.Tensor: 預測結果
        """
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x).view(-1)
                predictions.append(pred)

        # 堆疊所有預測結果
        stacked_preds = torch.stack(predictions)

        # 計算加權平均
        weighted_preds = torch.matmul(self.weights, stacked_preds)

        return weighted_preds

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        進行預測

        Args:
            x (torch.Tensor): 輸入特徵

        Returns:
            torch.Tensor: 二元預測結果
        """
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs >= self.threshold).float()
        return predictions

    @staticmethod
    def train_base_models(
        models: List[nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        device: torch.device,
        num_epochs: int,
        early_stopping_patience: int = 10,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[List[nn.Module], List[float]]:
        """
        訓練基礎模型

        Args:
            models (List[nn.Module]): 基礎模型列表
            train_loader (DataLoader): 訓練資料載入器
            val_loader (DataLoader): 驗證資料載入器
            criterion (nn.Module): 損失函數
            optimizers (List[torch.optim.Optimizer]): 優化器列表
            device (torch.device): 運算設備
            num_epochs (int): 訓練輪數
            early_stopping_patience (int, optional): 早停耐心值。默認為10
            logger (Optional[logging.Logger], optional): 日誌記錄器

        Returns:
            Tuple[List[nn.Module], List[float]]: 訓練後的模型列表和對應的驗證F1分數
        """
        if logger is None:
            logger = logging.getLogger('StockModel')

        best_models = []
        best_f1_scores = []

        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            logger.info(f"\n訓練基礎模型 {i+1}/{len(models)}...")

            best_val_f1 = 0
            patience_counter = 0
            best_state_dict = None

            for epoch in range(num_epochs):
                # 訓練
                model.train()
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features).view(-1)
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()

                # 驗證
                model.eval()
                all_outputs = []
                all_labels = []
                val_loss = 0
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(
                            device), labels.to(device)
                        outputs = model(features).view(-1)
                        loss = criterion(outputs, labels.float())
                        val_loss += loss.item()
                        all_outputs.extend(outputs.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                # 計算F1分數
                all_outputs = np.array(all_outputs)
                all_labels = np.array(all_labels)
                predictions = (all_outputs >= 0.5).astype(float)

                tp = np.sum((predictions == 1) & (all_labels == 1))
                fp = np.sum((predictions == 1) & (all_labels == 0))
                fn = np.sum((predictions == 0) & (all_labels == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                val_f1 = 2 * (precision * recall) / (precision +
                                                     recall) if (precision + recall) > 0 else 0

                logger.info(
                    f"模型 {i+1} - Epoch {epoch+1}/{num_epochs} - "
                    f"Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}"
                )

                # 保存最佳模型
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state_dict = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 早停
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"模型 {i+1} 早停：{early_stopping_patience} 個 epoch 沒有改善")
                    break

            # 恢復最佳模型狀態
            model.load_state_dict(best_state_dict)
            best_models.append(model)
            best_f1_scores.append(best_val_f1)

        return best_models, best_f1_scores

    def set_threshold(self, threshold: float) -> None:
        """
        設置預測閾值

        Args:
            threshold (float): 新的閾值
        """
        self.threshold = threshold

    def get_weights(self) -> List[float]:
        """
        獲取模型權重

        Returns:
            List[float]: 權重列表
        """
        return self.weights.tolist()

    def set_weights(self, weights: List[float]) -> None:
        """
        設置模型權重

        Args:
            weights (List[float]): 新的權重列表
        """
        total_weight = sum(weights)
        self.weights = torch.tensor([w / total_weight for w in weights])
