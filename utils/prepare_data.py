"""
資料準備模組
"""
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils.data_preprocessor import DataPreprocessor
from utils.data_augmentor import DataAugmentor
import torch


def prepare_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
    preprocessor: Optional[DataPreprocessor] = None,
    use_smote: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DataPreprocessor, DataAugmentor]:
    """
    準備訓練資料，包括預處理和分割

    Args:
        data: 輸入資料
        test_size: 測試集比例
        random_state: 隨機種子
        logger: 日誌記錄器
        preprocessor: 資料預處理器
        use_smote: 是否使用 SMOTE

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DataPreprocessor, DataAugmentor]:
            - 訓練特徵
            - 訓練標籤
            - 測試特徵
            - 測試標籤
            - 預處理器
            - 資料增強器
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 檢查是否有 NaN 值
    if data.isna().any().any():
        logger.warning("資料中包含 NaN 值，將進行處理")

    # 初始化預處理器
    if preprocessor is None:
        preprocessor = DataPreprocessor()
        logger.info("初始化新的預處理器")

    # 分割特徵和標籤
    X = data.drop(columns=[preprocessor.label_column])
    y = data[preprocessor.label_column]

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 擬合預處理器
    preprocessor.fit(X_train)

    # 轉換資料
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # 使用 SMOTE 進行資料平衡
    if use_smote:
        logger.info("使用 SMOTE 進行資料平衡")
        logger.info(f"平衡前訓練集大小: {len(X_train)}")
        logger.info(f"平衡前正樣本數量: {sum(y_train == 1)}")
        logger.info(f"平衡前負樣本數量: {sum(y_train == 0)}")

        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        logger.info(f"平衡後訓練集大小: {len(X_train)}")
        logger.info(f"平衡後正樣本數量: {sum(y_train == 1)}")
        logger.info(f"平衡後負樣本數量: {sum(y_train == 0)}")

    # 初始化資料增強器
    augmentor = DataAugmentor(logger=logger)
    # 分析資料特性並調整噪聲水平
    stats = augmentor.analyze_data_characteristics(X_train)
    augmentor.adjust_noise_levels(stats)

    # 轉換為 PyTorch 張量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train.values.reshape(-1, 1))
    y_test = torch.FloatTensor(y_test.values.reshape(-1, 1))

    # 檢查維度
    logger.info(f"訓練集特徵維度: {X_train.shape}")
    logger.info(f"訓練集標籤維度: {y_train.shape}")
    logger.info(f"測試集特徵維度: {X_test.shape}")
    logger.info(f"測試集標籤維度: {y_test.shape}")

    # 確保維度匹配
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"訓練集特徵和標籤的樣本數不匹配: {X_train.shape[0]} != {y_train.shape[0]}")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"測試集特徵和標籤的樣本數不匹配: {X_test.shape[0]} != {y_test.shape[0]}")

    return X_train, y_train, X_test, y_test, preprocessor, augmentor
