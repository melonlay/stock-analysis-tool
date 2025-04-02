"""
資料增強工具
"""
import numpy as np
import torch
from typing import Tuple, Dict, Optional
import logging
from config.model_config import AUGMENTATION_CONFIG


class DataAugmentor:
    """資料增強器"""

    def __init__(self, config: Dict = None, logger: Optional[logging.Logger] = None):
        """
        初始化資料增強器

        Args:
            config: 設定檔字典，如果為 None 則使用預設設定
            logger: 日誌記錄器
        """
        self.config = config or AUGMENTATION_CONFIG
        self.logger = logger or logging.getLogger(__name__)

        # 從設定檔載入參數
        self.noise_level = self.config['noise_level']
        self.jitter_level = self.config['jitter_level']
        self.rotation_level = self.config['rotation_level']
        self.scaling_level = self.config['scaling_level']
        self.trend_level = self.config['trend_level']
        self.seasonal_level = self.config['seasonal_level']
        self.time_warp_level = self.config['time_warp_level']
        self.mixup_alpha = self.config['mixup_alpha']

        # 自適應噪聲設定
        self.adaptive_noise_config = self.config['adaptive_noise']

        # 資料特徵統計
        self.feature_stats = {}

    def analyze_data_characteristics(self, X: np.ndarray) -> Dict:
        """
        分析資料特徵

        Args:
            X: 輸入特徵矩陣 (numpy array)

        Returns:
            包含資料特徵統計的字典
        """
        n_features = X.shape[1]
        stats = {}

        for i in range(n_features):
            feature = X[:, i]  # 直接使用 numpy array
            stats[i] = {
                'mean': float(np.mean(feature)),
                'std': float(np.std(feature)),
                'skewness': self._calculate_skewness(feature),
                'kurtosis': self._calculate_kurtosis(feature),
                'outlier_ratio': self._calculate_outlier_ratio(feature),
                'trend': self._calculate_trend(feature),
                'seasonality': self._calculate_seasonality(feature)
            }

        self.feature_stats = stats
        return stats

    def adjust_noise_levels(self, stats: Dict) -> None:
        """
        根據資料特性調整噪聲水平

        Args:
            stats: 資料特性統計字典
        """
        # 計算平均統計值
        mean_stats = {
            'std': np.mean([feature_stats['std'] for feature_stats in stats.values()]),
            'skewness': np.mean([feature_stats['skewness'] for feature_stats in stats.values()]),
            'kurtosis': np.mean([feature_stats['kurtosis'] for feature_stats in stats.values()]),
            'outlier_ratio': np.mean([feature_stats['outlier_ratio'] for feature_stats in stats.values()]),
            'trend': np.mean([feature_stats['trend'] for feature_stats in stats.values()]),
            'seasonality': np.mean([feature_stats['seasonality'] for feature_stats in stats.values()])
        }

        # 根據統計特性調整噪聲水平
        self.noise_level = self.adaptive_noise_config['base_noise'] * (
            1 +
            abs(mean_stats['skewness']) * self.adaptive_noise_config['skewness_factor'] +
            abs(mean_stats['kurtosis']) * self.adaptive_noise_config['kurtosis_factor'] +
            mean_stats['outlier_ratio'] * self.adaptive_noise_config['outlier_factor'] +
            abs(mean_stats['trend']) * self.adaptive_noise_config['trend_factor'] +
            mean_stats['seasonality'] *
            self.adaptive_noise_config['seasonal_factor']
        )

        # 確保噪聲水平在合理範圍內
        self.noise_level = min(max(self.noise_level, 0.01), 0.2)

    def add_adaptive_noise(self, X: torch.Tensor) -> torch.Tensor:
        """
        添加自適應噪聲

        Args:
            X: 輸入特徵矩陣 (PyTorch tensor)

        Returns:
            添加噪聲後的特徵矩陣 (PyTorch tensor)
        """
        X_noisy = X.clone()
        n_samples, n_features = X.shape
        device = X.device

        for i in range(n_features):
            feature_stats = self.feature_stats[i]

            # 生成基礎噪聲
            base_noise = torch.randn(
                n_samples, device=device) * self.noise_level

            # 根據偏度調整噪聲分布
            if feature_stats['skewness'] > 0:
                base_noise = torch.abs(base_noise)
            elif feature_stats['skewness'] < 0:
                base_noise = -torch.abs(base_noise)

            # 根據峰度調整極端值
            if feature_stats['kurtosis'] > 0:
                extreme_mask = torch.rand(
                    n_samples, device=device) < feature_stats['kurtosis'] * 0.1
                base_noise[extreme_mask] *= 2

            # 根據異常值比例調整噪聲強度
            base_noise *= feature_stats['outlier_ratio']

            # 添加趨勢噪聲
            trend_noise = torch.linspace(
                0, feature_stats['trend'] * self.trend_level, n_samples, device=device)

            # 添加季節性噪聲
            seasonal_noise = torch.sin(torch.linspace(0, 2*torch.pi, n_samples, device=device)) * \
                feature_stats['seasonality'] * self.seasonal_level

            # 組合所有噪聲
            total_noise = base_noise + trend_noise + seasonal_noise

            # 應用噪聲
            X_noisy[:, i] += total_noise

        return X_noisy

    def augment_single_sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        對單個樣本進行增強

        Args:
            x: 單個樣本的特徵向量 (PyTorch tensor)

        Returns:
            增強後的樣本 (PyTorch tensor)
        """
        # 確保輸入是二維的
        x = x.unsqueeze(0)
        return self.add_adaptive_noise(x).squeeze(0)

    def augment_batch(self, X: torch.Tensor, y: torch.Tensor, positive_only: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        對整個批次進行增強

        Args:
            X: 批次特徵矩陣 (PyTorch tensor)
            y: 批次標籤向量 (PyTorch tensor)
            positive_only: 是否只對正樣本進行增強

        Returns:
            增強後的特徵矩陣和標籤向量 (PyTorch tensor)
        """
        n_samples = len(X)
        X_aug = X.clone()
        y_aug = y.clone()

        for i in range(n_samples):
            if not positive_only or y[i] == 1:
                X_aug[i] = self.augment_single_sample(X[i])
            else:
                X_aug[i] = X[i]

        return X_aug, y_aug

    def augment_data(self, X: torch.Tensor, y: torch.Tensor, n_augment: int = 1, positive_only: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        對資料進行增強（用於離線增強）

        Args:
            X: 輸入特徵矩陣 (PyTorch tensor)
            y: 標籤向量 (PyTorch tensor)
            n_augment: 每個樣本要增強幾次
            positive_only: 是否只對正樣本進行增強

        Returns:
            增強後的特徵矩陣和標籤向量 (PyTorch tensor)
        """
        # 分析資料特徵
        stats = self.analyze_data_characteristics(X.cpu().numpy())

        # 調整噪聲水平
        self.adjust_noise_levels(stats)

        if positive_only:
            # 只對正樣本進行增強
            positive_mask = y == 1
            X_positive = X[positive_mask]
            y_positive = y[positive_mask]

            # 對正樣本進行增強
            n_positive = len(X_positive)
            X_aug = torch.zeros(
                (n_positive * n_augment, X.shape[1]), device=X.device)
            y_aug = torch.zeros(n_positive * n_augment, device=y.device)

            for i in range(n_positive):
                for j in range(n_augment):
                    idx = i * n_augment + j
                    X_aug[idx] = self.augment_single_sample(X_positive[i])
                    y_aug[idx] = y_positive[i]

            # 合併原始資料和增強後的資料
            X_augmented = torch.cat([X, X_aug], dim=0)
            y_augmented = torch.cat([y, y_aug], dim=0)

            return X_augmented, y_augmented
        else:
            # 對所有樣本進行增強
            n_samples = len(X)
            X_aug = torch.zeros(
                (n_samples * n_augment, X.shape[1]), device=X.device)
            y_aug = torch.zeros(n_samples * n_augment, device=y.device)

            for i in range(n_samples):
                for j in range(n_augment):
                    idx = i * n_augment + j
                    X_aug[idx] = self.augment_single_sample(X[i])
                    y_aug[idx] = y[i]

            # 合併原始資料和增強後的資料
            X_augmented = torch.cat([X, X_aug], dim=0)
            y_augmented = torch.cat([y, y_aug], dim=0)

            return X_augmented, y_augmented

    def _calculate_skewness(self, x: np.ndarray) -> float:
        """計算偏度"""
        return np.mean(((x - np.mean(x)) / np.std(x)) ** 3)

    def _calculate_kurtosis(self, x: np.ndarray) -> float:
        """計算峰度"""
        return np.mean(((x - np.mean(x)) / np.std(x)) ** 4) - 3

    def _calculate_outlier_ratio(self, x: np.ndarray) -> float:
        """計算異常值比例"""
        z_scores = np.abs((x - np.mean(x)) / np.std(x))
        return np.mean(z_scores > 2)

    def _calculate_trend(self, x: np.ndarray) -> float:
        """計算趨勢強度"""
        return np.polyfit(range(len(x)), x, 1)[0]

    def _calculate_seasonality(self, x: np.ndarray) -> float:
        """計算季節性強度"""
        fft = np.fft.fft(x)
        return np.max(np.abs(fft[1:len(fft)//2])) / len(x)
