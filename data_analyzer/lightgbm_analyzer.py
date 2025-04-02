"""
LightGBM 特徵重要性分析
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from .base import BaseAnalyzer


class LightGBMAnalyzer(BaseAnalyzer):
    """使用 LightGBM 分析特徵重要性"""

    def analyze(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        使用 LightGBM 分析特徵重要性

        Args:
            X (pd.DataFrame): 特徵數據框
            y (np.ndarray): 目標變數

        Returns:
            pd.DataFrame: 包含特徵重要性的數據框
        """
        try:
            # 嘗試使用 GPU
            lgb = LGBMClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                device='gpu',  # 使用 GPU
                gpu_platform_id=0,  # 使用第一個 GPU 平台
                gpu_device_id=0  # 使用第一個 GPU 設備
            )
            lgb.fit(X, y)
            print("成功使用 GPU 加速 LightGBM")
        except Exception as e:
            print(f"無法使用 GPU 加速 LightGBM（{str(e)}），切換回 CPU 模式")
            lgb = LGBMClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            lgb.fit(X, y)

        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': lgb.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
