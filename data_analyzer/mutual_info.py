"""
互信息特徵重要性分析
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from .base import BaseAnalyzer


class MutualInfoAnalyzer(BaseAnalyzer):
    """使用互信息分析特徵重要性"""

    def analyze(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        使用互信息分析特徵重要性

        Args:
            X (pd.DataFrame): 特徵數據框
            y (np.ndarray): 目標變數

        Returns:
            pd.DataFrame: 包含特徵重要性的數據框
        """
        importance_scores = mutual_info_classif(
            X, y,
            random_state=self.random_state,
            n_neighbors=3
        )

        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores
        })
        return importance.sort_values('importance', ascending=False)
