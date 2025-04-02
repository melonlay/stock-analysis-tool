"""
資料平衡模組
"""
import logging
from typing import Tuple, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) 實現

    用於處理不平衡資料集，通過在少數類別樣本之間生成合成樣本來平衡資料集。
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化SMOTE

        Args:
            k_neighbors (int, optional): 用於生成合成樣本的最近鄰數量。默認為5
            random_state (Optional[int], optional): 隨機種子。默認為None
            logger (Optional[logging.Logger], optional): 日誌記錄器。默認為None
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.logger = logger if logger else logging.getLogger(__name__)

        if random_state is not None:
            np.random.seed(random_state)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        對資料進行重採樣

        Args:
            X (np.ndarray): 特徵矩陣，形狀為 (n_samples, n_features)
            y (np.ndarray): 標籤數組，形狀為 (n_samples,)
            sampling_strategy (float, optional): 採樣策略，表示少數類別相對於多數類別的比例。
                                               默認為1.0，表示完全平衡

        Returns:
            Tuple[np.ndarray, np.ndarray]: 重採樣後的特徵矩陣和標籤數組
        """
        if len(X) != len(y):
            raise ValueError("特徵矩陣和標籤數組的長度不匹配")

        # 獲取少數類別和多數類別的索引
        minority_class = 1  # 假設1是少數類別
        majority_class = 0  # 假設0是多數類別

        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]

        n_minority = len(minority_indices)
        n_majority = len(majority_indices)

        self.logger.info(f"原始資料集中的少數類別樣本數: {n_minority}")
        self.logger.info(f"原始資料集中的多數類別樣本數: {n_majority}")

        if n_minority >= n_majority:
            self.logger.warning("少數類別樣本數大於或等於多數類別樣本數，無需進行SMOTE")
            return X, y

        # 計算需要生成的合成樣本數量
        n_synthetic = int(sampling_strategy * n_majority) - n_minority
        self.logger.info(f"將生成 {n_synthetic} 個合成樣本")

        # 獲取少數類別樣本
        X_minority = X[minority_indices]

        # 找到k個最近鄰
        n_neighbors = min(self.k_neighbors + 1, len(X_minority))
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(X_minority)
        distances, indices = neigh.kneighbors(X_minority)

        # 生成合成樣本
        synthetic_samples = []
        num_samples = 0

        while num_samples < n_synthetic:
            for i in range(n_minority):
                if num_samples >= n_synthetic:
                    break

                # 隨機選擇一個最近鄰
                nn_index = np.random.randint(1, n_neighbors)
                selected_nn = indices[i, nn_index]

                # 計算差值
                diff = X_minority[selected_nn] - X_minority[i]

                # 生成隨機權重
                weight = np.random.random()

                # 生成合成樣本
                synthetic_sample = X_minority[i] + weight * diff
                synthetic_samples.append(synthetic_sample)
                num_samples += 1

        # 合併原始樣本和合成樣本
        X_resampled = np.vstack([
            X,
            np.array(synthetic_samples)
        ])
        y_resampled = np.hstack([
            y,
            np.array([minority_class] * n_synthetic)
        ])

        self.logger.info(f"重採樣後的資料集大小: {len(X_resampled)}")
        self.logger.info(
            f"重採樣後的少數類別樣本數: {np.sum(y_resampled == minority_class)}")
        self.logger.info(
            f"重採樣後的多數類別樣本數: {np.sum(y_resampled == majority_class)}")

        return X_resampled, y_resampled

    def get_params(self) -> dict:
        """
        獲取參數設置

        Returns:
            dict: 參數字典
        """
        return {
            'k_neighbors': self.k_neighbors,
            'random_state': self.random_state
        }

    def set_params(self, **params) -> 'SMOTE':
        """
        設置參數

        Returns:
            SMOTE: 當前實例
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


def balance_data(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 1.0,
    k_neighbors: int = 5,
    random_state: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 SMOTE 技術平衡資料集

    Args:
        X (np.ndarray): 特徵矩陣
        y (np.ndarray): 標籤數組
        sampling_strategy (float, optional): 採樣策略。默認為 1.0
        k_neighbors (int, optional): 最近鄰數量。默認為 5
        random_state (Optional[int], optional): 隨機種子。默認為 None
        logger (Optional[logging.Logger], optional): 日誌記錄器。默認為 None

    Returns:
        Tuple[np.ndarray, np.ndarray]: 平衡後的特徵矩陣和標籤數組
    """
    smote = SMOTE(
        k_neighbors=k_neighbors,
        random_state=random_state,
        logger=logger
    )
    return smote.fit_resample(X, y, sampling_strategy=sampling_strategy)
