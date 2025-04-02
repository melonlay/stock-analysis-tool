# 股票分析機器學習模型 | Stock Analysis Machine Learning Model

[English](#english) | [中文](#chinese)

<a name="english"></a>
# Stock Analysis Machine Learning Model

A comprehensive machine learning project for stock analysis and prediction, implemented with PyTorch and scikit-learn. This project includes data preprocessing, feature engineering, model training, and prediction capabilities.

## Features

- **Data Preprocessing**
  - Standardization and normalization
  - Missing value handling
  - Outlier detection
  - Feature selection and engineering
  
- **Data Analysis**
  - Multiple analysis methods (Random Forest, XGBoost, LightGBM)
  - Feature importance analysis
  - Mutual information analysis
  - Visualization tools

- **Model Training**
  - Deep Neural Network (DNN) implementation with anti-overfitting features
  - Residual connections
  - Adaptive data augmentation
  - Advanced regularization techniques
  - Cross-validation
  - Early stopping and learning rate scheduling
  
- **Prediction Pipeline**
  - Batch prediction support
  - Result analysis and visualization
  - Performance metrics calculation

## Project Structure

```
.
├── analyzer.py              # Main analysis script
├── predict.py              # Prediction script
├── config/
│   └── model_config.py     # Configuration settings
├── data_analyzer/          # Analysis tools
│   ├── base.py
│   ├── lightgbm_analyzer.py
│   ├── mutual_info.py
│   ├── random_forest.py
│   ├── visualization.py
│   └── xgboost_analyzer.py
├── model/
│   └── dnn.py             # DNN model implementation
├── trainer/
│   ├── dataset/          # Dataset handling
│   ├── ensemble.py      # Ensemble methods
│   ├── evaluator.py    # Model evaluation
│   └── train_model.py  # Training script
├── utils/
│   ├── csv_reader.py    # Data loading utilities
│   ├── data_augmentor.py # Data augmentation
│   ├── data_preprocessor.py # Data preprocessing
│   ├── data_balancer.py # Data balancing
│   ├── plot_utils.py   # Visualization utilities
│   └── prepare_data.py # Data preparation
├── requirements.txt    # Project dependencies
└── LICENSE            # Project license
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/melonlay/stock-analysis-tool.git
cd stock-analysis-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place your stock data CSV files in the `data/` directory
2. Ensure your data contains required features and label columns
3. Configure data settings in `config/model_config.py`

### Training
```bash
python trainer/train_model.py
```

### Analysis
```bash
python analyzer.py
```

### Prediction
```bash
python predict.py
```

## Configuration

The main configuration file `config/model_config.py` includes several configuration sections:

### Data Augmentation Configuration
```python
AUGMENTATION_CONFIG = {
    # Basic noise settings
    'noise_level': 0.05,      # Base noise level
    'jitter_level': 0.02,     # Jitter level
    'rotation_level': 0.1,    # Rotation level
    'scaling_level': 0.1,     # Scaling level
    'trend_level': 0.03,      # Trend noise level
    'seasonal_level': 0.02,   # Seasonal noise level
    'time_warp_level': 0.1,   # Time warping level
    'mixup_alpha': 0.2,       # Mixup parameter

    # Adaptive noise settings
    'adaptive_noise': {
        'base_noise': 0.05,    # Base noise multiplier
        'skewness_factor': 0.5,# Skewness impact factor
        'kurtosis_factor': 0.3,# Kurtosis impact factor
        'outlier_factor': 0.4, # Outlier impact factor
        'trend_factor': 0.6,   # Trend impact factor
        'seasonal_factor': 0.4 # Seasonal impact factor
    },

    # Data analysis parameters
    'data_analysis': {
        'outlier_iqr_multiplier': 1.5,  # IQR outlier detection multiplier
        'trend_fit_degree': 1,          # Trend fitting degree
        'seasonality_fft_component': 1  # Seasonality FFT component
    }
}
```

### Training Configuration
```python
TRAINING_CONFIG = {
    # Data processing settings
    'batch_size': 32,                    # Batch size
    'use_augmentation': True,            # Enable data augmentation
    'n_augment': 3,                      # Number of augmentations per sample
    'positive_only_augment': True,       # Only augment positive samples
    
    # Model architecture settings
    'hidden_sizes': [128, 64, 32],       # Hidden layer sizes
    'dropout_rate': 0.5,                 # Dropout rate
    'weight_decay': 0.1,                 # Weight decay rate
    'pos_weight_multiplier': 2.0,        # Positive sample weight multiplier
    'use_residual': True,                # Use residual connections
    'l1_lambda': 0.05,                   # L1 regularization coefficient
    'gradient_clip': 0.5,                # Gradient clipping threshold
    'batch_norm': True,                  # Use batch normalization
    'activation': 'leaky_relu',          # Activation function
    
    # Optimizer settings
    'learning_rate': 0.00001,            # Learning rate
    'epochs': 100,                        # Number of epochs
    'patience': 15,                       # Early stopping patience
    
    # Learning rate scheduler settings
    'lr_scheduler': {
        'factor': 0.5,                   # Learning rate adjustment factor
        'patience': 10,                   # Learning rate adjustment patience
        'min_lr': 1e-7,                   # Minimum learning rate
    },

    # Weight initialization settings
    'weight_init': {
        'method': 'kaiming',             # Weight initialization method
        'mode': 'fan_out',               # Kaiming initialization mode
        'nonlinearity': 'leaky_relu'     # Nonlinearity type
    }
}
```

### Preprocessing Configuration
```python
PREPROCESSING_CONFIG = {
    'standardize': True,      # Enable standardization
    'handle_missing': True,   # Handle missing values
    'outlier_detection': {    # Outlier detection settings
        'method': 'zscore',
        'threshold': 3.0
    },
    'feature_selection': True,# Enable feature selection
    'label_column': '飆股',   # Label column name
    'exclude_columns': ['ID']# Columns to exclude
}
```

### Other Configurations
- **Validation Configuration**: Controls test split size, random state, and stratification
- **Logging Configuration**: Manages logging levels, formats, and visualization options
- **Save Configuration**: Controls model checkpoint and saving behavior

## Anti-Overfitting Features

The model includes several features to prevent overfitting:

1. **Architecture Optimizations**:
   - Residual connections for better gradient flow
   - Batch normalization for stable training
   - LeakyReLU activation for better gradient propagation
   - Reduced model capacity with smaller hidden layers

2. **Regularization Techniques**:
   - Dropout (0.5 rate) for feature co-adaptation prevention
   - L1 regularization (0.05 lambda) for sparsity
   - Weight decay (0.1) for parameter shrinkage
   - Gradient clipping (0.5 threshold) for stability

3. **Training Strategies**:
   - Smaller batch size (32) for better generalization
   - Lower learning rate (0.00001) for stable training
   - Longer training duration (100 epochs) with early stopping
   - Adaptive learning rate scheduling

4. **Data Augmentation**:
   - Adaptive noise based on data statistics
   - Positive sample oversampling
   - Mixup for better generalization
   - Trend and seasonal noise injection

## Important Notes

1. **Data Requirements**:
   - CSV format with proper headers
   - Consistent feature columns

2. **Performance Optimization**:
   - GPU acceleration supported
   - Memory-efficient data loading
   - Batch processing for large datasets

3. **Model Saving**:
   - Models are saved in the `output/` directory
   - Preprocessor state is saved for consistency
   - Training logs are generated automatically

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3). See [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice, and the authors are not responsible for any financial decisions made using this tool.

---

<a name="chinese"></a>
# 股票分析機器學習模型

這是一個完整的股票分析和預測機器學習專案，使用 PyTorch 和 scikit-learn 實作。本專案包含資料預處理、特徵工程、模型訓練和預測功能。

## 功能特點

- **資料預處理**
  - 標準化和正規化
  - 缺失值處理
  - 異常值檢測
  - 特徵選擇和工程

- **資料分析**
  - 多種分析方法（隨機森林、XGBoost、LightGBM）
  - 特徵重要性分析
  - 互信息分析
  - 視覺化工具

- **模型訓練**
  - 深度神經網路（DNN）實作，具備防過擬合特性
  - 殘差連接
  - 自適應資料增強
  - 進階正規化技術
  - 交叉驗證
  - 早停和學習率調度

- **預測流程**
  - 支援批次預測
  - 結果分析和視覺化
  - 效能指標計算

## 專案結構

```
.
├── analyzer.py              # 主要分析腳本
├── predict.py              # 預測腳本
├── config/
│   └── model_config.py     # 配置設定
├── data_analyzer/          # 分析工具
│   ├── base.py
│   ├── lightgbm_analyzer.py
│   ├── mutual_info.py
│   ├── random_forest.py
│   ├── visualization.py
│   └── xgboost_analyzer.py
├── model/
│   └── dnn.py             # DNN 模型實作
├── trainer/
│   ├── dataset/          # 資料集處理
│   ├── ensemble.py      # 集成方法
│   ├── evaluator.py    # 模型評估
│   └── train_model.py  # 訓練腳本
├── utils/
│   ├── csv_reader.py    # 資料載入工具
│   ├── data_augmentor.py # 資料增強
│   ├── data_preprocessor.py # 資料預處理
│   ├── data_balancer.py # 資料平衡
│   ├── plot_utils.py   # 視覺化工具
│   └── prepare_data.py # 資料準備
├── requirements.txt    # 專案依賴
└── LICENSE            # 專案授權
```

## 安裝

1. 複製專案：
```bash
git clone https://github.com/melonlay/stock-analysis-tool.git
cd stock-analysis-tool
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

### 資料準備
1. 將股票資料 CSV 檔案放在 `data/` 目錄中
2. 確保資料包含所需特徵和標籤欄位
3. 在 `config/model_config.py` 中配置資料設定

### 訓練
```bash
python trainer/train_model.py
```

### 分析
```bash
python analyzer.py
```

### 預測
```bash
python predict.py
```

## 配置說明

主要配置文件 `config/model_config.py` 包含以下幾個部分：

### 資料增強配置
```python
AUGMENTATION_CONFIG = {
    # 基本噪聲設定
    'noise_level': 0.05,      # 基礎噪聲水平
    'jitter_level': 0.02,     # 抖動水平
    'rotation_level': 0.1,    # 旋轉水平
    'scaling_level': 0.1,     # 縮放水平
    'trend_level': 0.03,      # 趨勢噪聲水平
    'seasonal_level': 0.02,   # 季節性噪聲水平
    'time_warp_level': 0.1,   # 時間扭曲水平
    'mixup_alpha': 0.2,       # Mixup 混合參數

    # 自適應噪聲設定
    'adaptive_noise': {
        'base_noise': 0.05,    # 基礎噪聲倍數
        'skewness_factor': 0.5,# 偏度影響因子
        'kurtosis_factor': 0.3,# 峰度影響因子
        'outlier_factor': 0.4, # 異常值影響因子
        'trend_factor': 0.6,   # 趨勢影響因子
        'seasonal_factor': 0.4 # 季節性影響因子
    },

    # 資料分析參數
    'data_analysis': {
        'outlier_iqr_multiplier': 1.5,  # IQR 異常值檢測乘數
        'trend_fit_degree': 1,          # 趨勢擬合次數
        'seasonality_fft_component': 1  # 季節性 FFT 分量
    }
}
```

### 訓練配置
```python
TRAINING_CONFIG = {
    # 資料處理設定
    'batch_size': 32,                    # 批次大小
    'use_augmentation': True,            # 是否使用資料增強
    'n_augment': 3,                      # 每個樣本增強次數
    'positive_only_augment': True,       # 是否只對正樣本進行增強
    
    # 模型架構設定
    'hidden_sizes': [128, 64, 32],       # 隱藏層大小
    'dropout_rate': 0.5,                 # Dropout 率
    'weight_decay': 0.1,                 # 權重衰減率
    'pos_weight_multiplier': 2.0,        # 正樣本權重倍數
    'use_residual': True,                # 是否使用殘差連接
    'l1_lambda': 0.05,                   # L1 正規化係數
    'gradient_clip': 0.5,                # 梯度裁剪閾值
    'batch_norm': True,                  # 是否使用批次正規化
    'activation': 'leaky_relu',          # 激活函數
    
    # 優化器設定
    'learning_rate': 0.00001,            # 學習率
    'epochs': 100,                        # 訓練輪數
    'patience': 15,                       # 早停耐心值
    
    # 學習率排程器設定
    'lr_scheduler': {
        'factor': 0.5,                   # 學習率調整因子
        'patience': 10,                   # 學習率調整耐心值
        'min_lr': 1e-7,                   # 最小學習率
    },

    # 權重初始化設定
    'weight_init': {
        'method': 'kaiming',             # 權重初始化方法
        'mode': 'fan_out',               # Kaiming 初始化模式
        'nonlinearity': 'leaky_relu'     # 非線性函數類型
    }
}
```

### 預處理配置
```python
PREPROCESSING_CONFIG = {
    'standardize': True,      # 是否進行標準化
    'handle_missing': True,   # 是否處理缺失值
    'outlier_detection': {    # 異常值檢測設定
        'method': 'zscore',
        'threshold': 3.0
    },
    'feature_selection': True,# 是否進行特徵選擇
    'label_column': '飆股',   # 標籤欄位名稱
    'exclude_columns': ['ID']# 要排除的欄位
}
```

### 其他配置
- **驗證配置**：控制測試集分割比例、隨機種子和分層抽樣
- **日誌配置**：管理日誌等級、格式和視覺化選項
- **儲存配置**：控制模型檢查點和儲存行為

## 防過擬合特性

模型包含多個防止過擬合的特性：

1. **架構優化**：
   - 殘差連接以改善梯度流動
   - 批次正規化以穩定訓練
   - LeakyReLU 激活函數以改善梯度傳播
   - 較小的隱藏層以減少模型容量

2. **正規化技術**：
   - Dropout（0.5 比率）防止特徵共適應
   - L1 正規化（0.05 係數）促進稀疏性
   - 權重衰減（0.1）收縮參數
   - 梯度裁剪（0.5 閾值）穩定訓練

3. **訓練策略**：
   - 較小的批次大小（32）改善泛化能力
   - 較低的學習率（0.00001）穩定訓練
   - 較長的訓練時間（100 輪）配合早停
   - 自適應學習率調度

4. **資料增強**：
   - 基於資料統計的自適應噪聲
   - 正樣本過採樣
   - Mixup 改善泛化能力
   - 趨勢和季節性噪聲注入

## 重要注意事項

1. **資料要求**：
   - CSV 格式且具有適當的表頭
   - 一致的特徵欄位

2. **效能優化**：
   - 支援 GPU 加速
   - 記憶體效率的資料載入
   - 大型資料集的批次處理

3. **模型儲存**：
   - 模型儲存在 `output/` 目錄
   - 預處理器狀態會被保存以確保一致性
   - 自動生成訓練日誌

## 授權

本專案採用 GNU General Public License v3.0 (GPL-3) 授權。詳見 [LICENSE](LICENSE) 檔案。

## 免責聲明

本軟體僅供教育和研究用途。不構成財務建議，作者不對使用本工具做出的財務決策負責。