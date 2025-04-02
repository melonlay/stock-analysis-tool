# 股票分析機器學習模型 | Stock Analysis Machine Learning Model

[English](#english) | [中文](#chinese)

<a name="english"></a>
# Stock Analysis Machine Learning Model

A machine learning project for stock analysis, implemented with PyTorch for deep learning models, including data preprocessing and data augmentation features.

## Features

- Data Preprocessing: Standardization, missing value handling, outlier detection, feature selection
- Data Augmentation: Adaptive noise, time series transformation, Mixup
- Deep Learning Model: DNN architecture with batch normalization and Dropout
- Model Training: Support for early stopping, learning rate adjustment, weight decay, and other optimization techniques
- Complete logging and model saving functionality

## Project Structure

```
.
├── config/
│   └── model_config.py    # Model and training configuration
├── trainer/
│   └── train_model.py     # Main training script
├── utils/
│   ├── data_augmentor.py  # Data augmentation tools
│   └── data_preprocessor.py # Data preprocessing tools
├── .gitignore            # Git ignore file
└── README.md            # Project documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare Data:
   - Place data files in the `data/` directory
   - Data should be in CSV format with feature and label columns

2. Train Model:
   ```bash
   python trainer/train_model.py --data_dir data/ --output_dir output/
   ```

3. Optional Parameters:
   - `--batch_size`: Batch size
   - `--learning_rate`: Learning rate
   - `--epochs`: Number of training epochs
   - `--use_augmentation`: Whether to use data augmentation
   - `--save_preprocessor`: Whether to save preprocessor state

## Configuration

Main configuration file is located at `config/model_config.py`, including:

- Data augmentation settings
- Model training settings
- Data preprocessing settings
- Validation settings
- Logging settings
- Model saving settings

## Notes

- Ensure data file format is correct
- GPU is recommended for training
- Adjust configuration parameters as needed

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3). This means:

- You are free to use, modify, and distribute this program
- You must provide the source code when distributing
- Modified versions must also use the same license
- You must include a complete copy of the license when distributing

See the [LICENSE](LICENSE) file for the full license text.

## Disclaimer

This program is for research and learning purposes only and does not constitute any investment advice. Users bear all risks when using this program for actual investment decisions.

## Contact

For any questions or suggestions, please contact:

- GitHub: [stock-analysis-tool](https://github.com/melonlay/stock-analysis-tool.git)

---

<a name="chinese"></a>
# 股票分析機器學習模型

這是一個使用 PyTorch 實作的股票分析機器學習模型，用於預測股票價格走勢。

## 功能特點

- 使用 PyTorch 實作深度學習模型
- 支援多種資料預處理方法
- 提供完整的模型訓練和評估流程
- 包含詳細的日誌記錄和模型保存功能

## 專案結構

```
.
├── config/
│   └── model_config.py      # 模型配置檔案
├── data/                    # 資料目錄
├── models/                  # 模型定義
│   ├── __init__.py
│   └── model.py
├── trainer/                 # 訓練相關程式碼
│   ├── __init__.py
│   └── train_model.py
├── utils/                   # 工具函數
│   ├── __init__.py
│   ├── data_augmentor.py
│   └── data_preprocessor.py
├── requirements.txt         # 專案依賴
├── README.md               # 專案說明文件
└── LICENSE                 # GPL-3 授權條款
```

## 安裝需求

```bash
pip install -r requirements.txt
```

## 使用方法

1. 準備資料：
   - 將股票資料放在 `data` 目錄下
   - 資料格式應為 CSV 檔案

2. 配置模型：
   - 在 `config/model_config.py` 中調整模型參數
   - 可以修改資料預處理和訓練相關的設定

3. 訓練模型：
   ```bash
   python trainer/train_model.py
   ```

4. 評估結果：
   - 訓練完成後，模型會自動保存
   - 可以在日誌中查看訓練過程和結果

## 配置說明

### 資料預處理配置

```python
PREPROCESSING_CONFIG = {
    'label_column': '飆股',  # 標籤欄位名稱
    'standardize': True,    # 是否進行標準化
    'handle_missing': True, # 是否處理缺失值
    'detect_outliers': True,# 是否檢測異常值
    'feature_selection': True # 是否進行特徵選擇
}
```

### 模型配置

```python
MODEL_CONFIG = {
    'input_size': 10,       # 輸入特徵數量
    'hidden_size': 64,      # 隱藏層大小
    'num_layers': 2,        # LSTM 層數
    'output_size': 1,       # 輸出大小
    'dropout': 0.2,         # Dropout 比率
    'learning_rate': 0.001, # 學習率
    'batch_size': 32,       # 批次大小
    'num_epochs': 100       # 訓練輪數
}
```

## 重要說明

1. 資料格式：
   - 輸入資料應為 CSV 格式
   - 必須包含指定的標籤欄位
   - 建議進行資料預處理以提高模型效果

2. 模型訓練：
   - 訓練過程會自動保存最佳模型
   - 可以通過日誌追蹤訓練進度
   - 支援早停機制避免過擬合

3. 效能優化：
   - 使用 GPU 加速訓練（如果可用）
   - 支援批次處理提高效率
   - 提供記憶體優化選項

## 授權說明

本專案採用 GNU General Public License v3.0 (GPL-3) 授權條款。這表示：

- 您可以自由使用、修改和分發本程式
- 您必須在分發時提供原始碼
- 修改後的版本也必須使用相同的授權條款
- 您必須在分發時附上完整的授權條款副本

詳細的授權條款請參見 [LICENSE](LICENSE) 檔案。

## 免責聲明

本程式僅供研究和學習使用，不構成任何投資建議。使用本程式進行實際投資決策的風險由使用者自行承擔。

## 聯絡方式

如有任何問題或建議，請通過以下方式聯絡：

- GitHub: [stock-analysis-tool](https://github.com/melonlay/stock-analysis-tool.git)