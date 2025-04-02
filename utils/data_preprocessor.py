"""
資料預處理模組
"""
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import joblib
import os
from sklearn.model_selection import train_test_split
from config.model_config import PREPROCESSING_CONFIG


class DataPreprocessor:
    """資料預處理器類別"""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        standardize: bool = True,
        handle_missing: bool = True,
        detect_outliers: bool = True,
        feature_selection: bool = True
    ):
        """
        初始化預處理器

        Args:
            logger (Optional[logging.Logger]): 日誌記錄器
            standardize (bool): 是否進行標準化
            handle_missing (bool): 是否處理缺失值
            detect_outliers (bool): 是否檢測異常值
            feature_selection (bool): 是否進行特徵選擇
        """
        self.logger = logger or logging.getLogger('StockModel')
        self.standardize = standardize
        self.handle_missing = handle_missing
        self.detect_outliers = detect_outliers
        self.feature_selection = feature_selection
        self.label_column = PREPROCESSING_CONFIG['label_column']
        self.exclude_columns = PREPROCESSING_CONFIG['exclude_columns']

        self.categorical_columns = None
        self.numeric_columns = None
        self.knn_columns = None
        self.scaler = StandardScaler() if standardize else None
        self.imputer = None
        self.categorical_fill_values = {}  # 儲存類別型特徵的填補值
        self.numeric_fill_values = {}  # 儲存數值型特徵的填補值
        self.important_features = [
            # 技術指標
            '技術指標_月RSI(10)', '技術指標_季D(9)', '技術指標_季K(9)',
            '技術指標_Alpha(250D)', '技術指標_月ADX(14)', '技術指標_週RSI(10)',
            '技術指標_RSI(5)', '技術指標_週RSI(5)', '技術指標_ADX(14)',
            '技術指標_季DIF-季MACD', '技術指標_MACD', '技術指標_週ADX(14)',
            '技術指標_月RSI(5)', '技術指標_RSI(10)', '技術指標_月ADXR(14)',
            '技術指標_月D(9)', '技術指標_季ADX(14)', '技術指標_K(9)',
            '技術指標_乖離率(250日)', '技術指標_Beta係數(65D)',

            # 加權指數相關
            '上市加權指數前13天成交量', '上市加權指數前17天成交量',
            '上市加權指數20天報酬率', '上市加權指數10天成交量波動度',
            '上市加權指數前2天成交量', '上市加權指數5天成交量波動度',
            '上市加權指數前4天成交量', '上市加權指數前3天成交量',
            '上市加權指數成交量', '上市加權指數前12天成交量',
            '上市加權指數1天報酬率', '上市加權指數前20天成交量',
            '上市加權指數前10天成交量', '上市加權指數前5天成交量',
            '上市加權指數10天波動度', '上市加權指數19天乖離率',
            '上市加權指數前18天成交量', '上市加權指數20天成交量波動度',
            '上市加權指數前6天成交量', '上市加權指數5天乖離率',
            '上市加權指數10天乖離率', '上市加權指數前8天成交量',
            '上市加權指數前11天成交量', '上市加權指數前1天成交量',

            # 個股相關
            '個股5天報酬率', '個股10天乖離率', '個股5天乖離率',
            '個股券商分點區域分析_近60日成交5分點區域密度',

            # 外資相關
            '日外資_外資尚可投資張數'
        ]  # 重要特徵列表

    def fit(self, df: pd.DataFrame) -> None:
        """
        擬合預處理器

        Args:
            df (pd.DataFrame): 訓練資料（只包含特徵）
        """
        self.logger.info("開始擬合預處理器...")

        # 檢查原始資料是否有 NaN
        self.logger.info("步驟 1/5: 檢查原始資料中的 NaN...")
        self.logger.debug("檢查原始資料中的 NaN:")
        self.logger.debug(f"特徵中的 NaN 數量: {df.isna().sum().sum()}")

        # 分離數值型和類別型特徵
        self.logger.info("步驟 2/5: 分離數值型和類別型特徵...")
        columns_to_drop = [self.label_column] + self.exclude_columns
        X = df.drop(columns_to_drop, axis=1, errors='ignore')
        self.categorical_columns = X.select_dtypes(
            include=['object', 'category']).columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.logger.info(
            f"找到 {len(self.categorical_columns)} 個類別型特徵和 {len(self.numeric_columns)} 個數值型特徵")

        # 處理類別型特徵
        if self.handle_missing:
            self.logger.info("步驟 3/5: 處理類別型特徵...")
            for col in self.categorical_columns:
                missing_ratio = X[col].isna().mean()
                if missing_ratio > 0.5:  # 如果缺失值超過 50%
                    X[col] = X[col].fillna('missing')
                    self.categorical_fill_values[col] = 'missing'
                    self.logger.debug(f"特徵 {col} 使用 'missing' 類別填補")
                else:
                    fill_value = X[col].mode().iloc[0]
                    X[col] = X[col].fillna(fill_value)
                    self.categorical_fill_values[col] = fill_value
                    self.logger.debug(f"特徵 {col} 使用眾數填補: {fill_value}")

        # 處理數值型特徵
        if self.handle_missing:
            self.logger.info("步驟 4/5: 處理數值型特徵...")
            self.knn_columns = []
            for col in self.numeric_columns:
                missing_ratio = X[col].isna().mean()
                # 只對重要特徵且缺失值比例適中的特徵使用 KNN 填補
                if col in self.important_features and 0 < missing_ratio <= 0.3:
                    self.knn_columns.append(col)
                else:
                    fill_value = X[col].median()
                    X[col] = X[col].fillna(fill_value)
                    self.numeric_fill_values[col] = fill_value
                    self.logger.debug(f"特徵 {col} 使用中位數填補: {fill_value}")

            # 如果有需要 KNN 填補的欄位，擬合 KNN 填補器
            if self.knn_columns:
                self.logger.info(f"使用 KNN 填補 {len(self.knn_columns)} 個特徵...")
                self.imputer = KNNImputer(n_neighbors=5, weights='uniform')
                X[self.knn_columns] = self.imputer.fit_transform(
                    X[self.knn_columns])

        # 擬合標準化器
        if self.standardize:
            self.logger.info("步驟 5/5: 擬合標準化器...")
            self.scaler.fit(X)
            self.logger.info("預處理器擬合完成！")

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        轉換資料

        Args:
            df (pd.DataFrame): 要轉換的資料

        Returns:
            Tuple[np.ndarray, np.ndarray]: (特徵矩陣, 標籤陣列)
        """
        self.logger.info("開始轉換資料...")

        # 移除不需要的欄位
        self.logger.info("步驟 1/4: 準備資料...")
        columns_to_drop = [self.label_column] + self.exclude_columns
        X = df.drop(columns_to_drop, axis=1, errors='ignore')

        # 處理類別型特徵
        if self.handle_missing:
            self.logger.info("步驟 2/4: 處理類別型特徵...")
            for col in self.categorical_columns:
                X[col] = X[col].fillna(self.categorical_fill_values[col])

        # 處理數值型特徵
        if self.handle_missing:
            self.logger.info("步驟 3/4: 處理數值型特徵...")
            for col in self.numeric_columns:
                if col not in self.knn_columns:
                    X[col] = X[col].fillna(self.numeric_fill_values[col])

            # 使用 KNN 填補
            if self.knn_columns:
                self.logger.info(f"使用 KNN 填補 {len(self.knn_columns)} 個特徵...")
                X[self.knn_columns] = self.imputer.transform(
                    X[self.knn_columns])

        # 標準化特徵
        if self.standardize:
            self.logger.info("步驟 4/4: 標準化特徵...")
            X = self.scaler.transform(X)

        self.logger.info("資料轉換完成！")
        return X, df[self.label_column].values

    @staticmethod
    def transform_unlabeled(data: Union[pd.DataFrame, pd.Series, Dict[str, Any]], preprocessor_path: str) -> np.ndarray:
        """
        轉換未標記資料（單筆或小型資料集）

        Args:
            data (Union[pd.DataFrame, pd.Series, Dict[str, Any]]): 要轉換的資料，可以是 DataFrame、Series 或字典
            preprocessor_path (str): 預處理器狀態檔案的路徑

        Returns:
            np.ndarray: 轉換後的特徵矩陣
        """
        # 載入預處理器狀態
        state = joblib.load(preprocessor_path)

        # 將輸入轉換為 DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("輸入資料必須是 DataFrame、Series 或字典")

        # 移除不需要的欄位
        columns_to_drop = [state['label_column']] + state['exclude_columns']
        df = df.drop(columns_to_drop, axis=1, errors='ignore')

        # 確保所有必要的欄位都存在
        required_columns = list(
            state['categorical_columns']) + list(state['numeric_columns'])
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的欄位: {missing_columns}")

        # 處理類別型特徵
        for col in state['categorical_columns']:
            df[col] = df[col].fillna(state['categorical_fill_values'][col])

        # 處理數值型特徵
        for col in state['numeric_columns']:
            if col not in state['knn_columns']:
                df[col] = df[col].fillna(state['numeric_fill_values'][col])

        # 使用 KNN 填補
        if state['knn_columns']:
            df[state['knn_columns']] = state['imputer'].transform(
                df[state['knn_columns']])

        # 標準化特徵
        X = state['scaler'].transform(df)

        return X

    def save(self, save_dir: str) -> None:
        """
        保存預處理器狀態

        Args:
            save_dir (str): 保存目錄
        """
        state = {
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'knn_columns': self.knn_columns,
            'categorical_fill_values': self.categorical_fill_values,
            'numeric_fill_values': self.numeric_fill_values,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_column': self.label_column,
            'exclude_columns': self.exclude_columns
        }
        joblib.dump(state, save_dir)

    @staticmethod
    def load(preprocessor_path: str, logger: Optional[logging.Logger] = None) -> 'DataPreprocessor':
        """
        載入預處理器狀態

        Args:
            preprocessor_path (str): 預處理器狀態檔案的路徑
            logger (Optional[logging.Logger]): 日誌記錄器

        Returns:
            DataPreprocessor: 載入狀態後的預處理器實例
        """
        state = joblib.load(preprocessor_path)
        preprocessor = DataPreprocessor(logger=logger)
        preprocessor.categorical_columns = state['categorical_columns']
        preprocessor.numeric_columns = state['numeric_columns']
        preprocessor.knn_columns = state['knn_columns']
        preprocessor.categorical_fill_values = state['categorical_fill_values']
        preprocessor.numeric_fill_values = state['numeric_fill_values']
        preprocessor.scaler = state['scaler']
        preprocessor.imputer = state['imputer']
        preprocessor.label_column = state['label_column']
        preprocessor.exclude_columns = state['exclude_columns']
        return preprocessor
