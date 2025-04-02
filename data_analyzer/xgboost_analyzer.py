"""
XGBoost 特徵重要性分析
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from .base import BaseAnalyzer


class XGBoostAnalyzer(BaseAnalyzer):
    """使用 XGBoost 分析特徵重要性"""

    def analyze(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        使用 XGBoost 分析特徵重要性

        Args:
            X (pd.DataFrame): 特徵數據框
            y (np.ndarray): 目標變數

        Returns:
            pd.DataFrame: 包含特徵重要性的數據框
        """
        try:
            # 嘗試使用 GPU（使用新的參數格式）
            xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                tree_method='hist',  # 使用直方圖方法
                device='cuda'  # 指定使用 CUDA 設備
            )
            xgb.fit(X, y)
            print("成功使用 GPU 加速 XGBoost")
        except Exception as e:
            print(f"無法使用 GPU 加速 XGBoost（{str(e)}），切換回 CPU 模式")
            xgb = XGBClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            xgb.fit(X, y)

        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
