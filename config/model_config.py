"""
模型訓練和資料增強設定檔
"""
from typing import Dict, Any
import numpy as np

# 資料增強設定
AUGMENTATION_CONFIG: Dict[str, Any] = {
    # 基本噪聲設定
    'noise_level': 0.05,  # 基礎噪聲水平
    'jitter_level': 0.02,  # 抖動水平
    'rotation_level': 0.1,  # 旋轉水平
    'scaling_level': 0.1,  # 縮放水平
    'trend_level': 0.03,  # 趨勢噪聲水平
    'seasonal_level': 0.02,  # 季節性噪聲水平
    'time_warp_level': 0.1,  # 時間扭曲水平
    'mixup_alpha': 0.2,  # Mixup 混合參數

    # 自適應噪聲設定
    'adaptive_noise': {
        'base_noise': 0.05,  # 基礎噪聲倍數
        'skewness_factor': 0.5,  # 偏度影響因子
        'kurtosis_factor': 0.3,  # 峰度影響因子
        'outlier_factor': 0.4,  # 異常值影響因子
        'trend_factor': 0.6,  # 趨勢影響因子
        'seasonal_factor': 0.4,  # 季節性影響因子
    },

    # 資料分析參數
    'data_analysis': {
        'outlier_iqr_multiplier': 1.5,  # IQR 異常值檢測乘數
        'trend_fit_degree': 1,  # 趨勢擬合次數
        'seasonality_fft_component': 1,  # 季節性 FFT 分量
    }
}

# 模型訓練設定
TRAINING_CONFIG: Dict[str, Any] = {
    # 資料處理設定
    'batch_size': 64,  # 批次大小
    'use_augmentation': True,  # 是否使用資料增強
    'n_augment': 5,  # 每個樣本增強次數
    'positive_only_augment': True,  # 是否只對正樣本進行增強

    # 模型架構設定
    'hidden_sizes': [256, 128, 64],  # 隱藏層大小
    'dropout_rate': 0.4,  # Dropout 率
    'weight_decay': 0.02,  # 權重衰減率
    'pos_weight_multiplier': 1.5,  # 正樣本權重倍數

    # 優化器設定
    'learning_rate': 0.00005,  # 學習率
    'epochs': 50,  # 訓練輪數
    'patience': 10,  # 早停耐心值

    # 學習率排程器設定
    'lr_scheduler': {
        'factor': 0.7,  # 學習率調整因子
        'patience': 7,  # 學習率調整耐心值
        'min_lr': 1e-6,  # 最小學習率
    }
}

# 資料預處理設定
PREPROCESSING_CONFIG: Dict[str, Any] = {
    'standardize': True,  # 是否進行標準化
    'handle_missing': True,  # 是否處理缺失值
    'outlier_detection': True,  # 是否進行異常值檢測
    'feature_selection': True,  # 是否進行特徵選擇
    'label_column': '飆股',  # 標籤欄位名稱
    'exclude_columns': ['ID'],  # 要排除的欄位
    'outlier_detection': {
        'method': 'zscore',
        'threshold': 3.0
    }
}

# 驗證設定
VALIDATION_CONFIG: Dict[str, Any] = {
    'test_size': 0.2,  # 測試集比例
    'random_state': 42,  # 隨機種子
    'stratify': True,  # 是否進行分層抽樣
}

# 日誌設定
LOGGING_CONFIG: Dict[str, Any] = {
    'level': 'INFO',  # 日誌等級
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日誌格式
    'save_tensorboard': True,  # 是否保存 TensorBoard 日誌
    'save_plots': True,  # 是否保存訓練圖表
}

# 模型儲存設定
SAVE_CONFIG: Dict[str, Any] = {
    'save_best_model': True,  # 是否保存最佳模型
    'save_last_model': True,  # 是否保存最後一個模型
    'save_checkpoints': True,  # 是否保存檢查點
    'checkpoint_frequency': 5,  # 檢查點保存頻率
}
