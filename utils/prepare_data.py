"""
資料準備模組
"""
import logging
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .data_preprocessor import DataPreprocessor
from .data_balancer import SMOTE
from .data_augmentor import DataAugmentor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
    preprocessor: Optional[DataPreprocessor] = None,
    use_smote: bool = True,
    use_augmentation: bool = False,
    n_augment: int = 1,
    positive_only: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """
    準備訓練資料

    Args:
        df (pd.DataFrame): 原始資料框
        test_size (float, optional): 測試集比例。預設為 0.2
        random_state (int, optional): 隨機種子。預設為 42
        logger (Optional[logging.Logger]): 日誌記錄器
        preprocessor (Optional[DataPreprocessor]): 預處理器，如果提供則使用現有的
        use_smote (bool, optional): 是否使用 SMOTE 進行資料平衡。預設為 True
        use_augmentation (bool, optional): 是否使用資料增強。預設為 False
        n_augment (int, optional): 每個樣本要增強幾次。預設為 1
        positive_only (bool, optional): 是否只對正樣本進行增強。預設為 True

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]: 
            (X_train, X_test, y_train, y_test, preprocessor)
    """
    if logger is None:
        logger = logging.getLogger('StockModel')

    # 創建或使用現有的預處理器
    if preprocessor is None:
        preprocessor = DataPreprocessor(logger)
        preprocessor.fit(df)

    # 分割資料
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['飆股']
    )

    # 轉換資料
    X_train, y_train = preprocessor.transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)

    # 檢查處理後的資料是否有 NaN
    logger.debug("檢查處理後的資料中的 NaN:")
    logger.debug(f"訓練集特徵中的 NaN 數量: {np.isnan(X_train).sum()}")
    logger.debug(f"測試集特徵中的 NaN 數量: {np.isnan(X_test).sum()}")
    logger.debug(f"訓練集標籤中的 NaN 數量: {np.isnan(y_train).sum()}")
    logger.debug(f"測試集標籤中的 NaN 數量: {np.isnan(y_test).sum()}")

    # 使用 SMOTE 進行資料平衡
    if use_smote:
        logger.info("使用 SMOTE 進行資料平衡...")
        smote = SMOTE(random_state=random_state, logger=logger)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"平衡後的訓練集大小: {len(X_train)}")
        logger.info(f"平衡後的訓練集正樣本比例: {np.mean(y_train):.4f}")

    # 使用資料增強
    if use_augmentation:
        logger.info("使用資料增強...")
        augmentor = DataAugmentor(logger=logger)
        X_train, y_train = augmentor.augment_data(
            X_train, y_train, n_augment=n_augment, positive_only=positive_only
        )
        logger.info(f"增強後的訓練集大小: {len(X_train)}")
        logger.info(f"增強後的訓練集正樣本比例: {np.mean(y_train):.4f}")

    return X_train, X_test, y_train, y_test, preprocessor
