"""
隨機森林特徵重要性分析
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseAnalyzer


class RandomForestAnalyzer(BaseAnalyzer):
    """使用隨機森林分析特徵重要性"""

    def analyze(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        使用隨機森林分析特徵重要性

        Args:
            X (pd.DataFrame): 特徵數據框
            y (np.ndarray): 目標變數

        Returns:
            pd.DataFrame: 包含特徵重要性的數據框
        """
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)

        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)
