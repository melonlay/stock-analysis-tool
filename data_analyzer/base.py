"""
基礎類別和共用函數
"""
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class ImportanceMethod(str, Enum):
    """特徵重要性分析方法列舉"""
    RANDOM_FOREST = 'rf'
    XGBOOST = 'xgb'
    LIGHTGBM = 'lgb'
    MUTUAL_INFO = 'mi'


class BaseAnalyzer(ABC):
    """特徵重要性分析器基礎類別"""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        初始化分析器

        Args:
            n_estimators (int, optional): 決策樹數量。預設為 100
            random_state (int, optional): 隨機種子。預設為 42
        """
        self.n_estimators = n_estimators
        self.random_state = random_state

    @abstractmethod
    def analyze(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        分析特徵重要性

        Args:
            X (pd.DataFrame): 特徵數據框
            y (np.ndarray): 目標變數

        Returns:
            pd.DataFrame: 包含特徵重要性的數據框
        """
        pass


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    預處理特徵，將非數值型特徵轉換為數值型，並處理缺失值

    Args:
        X (pd.DataFrame): 原始特徵數據框

    Returns:
        pd.DataFrame: 預處理後的特徵數據框
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer

    # 複製數據框以避免修改原始數據
    X_processed = X.copy()

    # 對每個非數值型欄位進行編碼
    for column in X_processed.columns:
        if X_processed[column].dtype == 'object':
            le = LabelEncoder()
            # 將 NaN 轉換為特殊字符串
            X_processed[column] = X_processed[column].fillna('MISSING')
            X_processed[column] = le.fit_transform(
                X_processed[column].astype(str))
        else:
            # 對數值型特徵使用中位數填充
            imputer = SimpleImputer(strategy='median')
            X_processed[column] = imputer.fit_transform(
                X_processed[[column]])

    return X_processed
