"""
模型評估模組
"""
import logging
import os
import sys
import argparse
from datetime import datetime
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# 添加專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.csv_reader import read_csv_files
    from ..utils.data_preprocessor import DataPreprocessor
    from ..utils.prepare_data import prepare_data
    from ..model.dnn import create_dnn_model
    from .dataset.StockDataset import StockDataset
except ImportError:
    from utils.csv_reader import read_csv_files
    from utils.data_preprocessor import DataPreprocessor
    from utils.prepare_data import prepare_data
    from model.dnn import create_dnn_model
    from trainer.dataset.StockDataset import StockDataset


def setup_logger(log_dir: str) -> logging.Logger:
    """
    設定日誌系統

    Args:
        log_dir (str): 日誌保存目錄

    Returns:
        logging.Logger: 配置好的日誌記錄器
    """
    logger = logging.getLogger('StockModel')
    logger.setLevel(logging.DEBUG)

    # 創建日誌目錄
    os.makedirs(log_dir, exist_ok=True)

    # 檔案處理器
    log_file = os.path.join(log_dir, 'evaluation.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 設定格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加處理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float, float, float]:
    """
    評估模型

    Args:
        model (nn.Module): 模型
        data_loader (DataLoader): 資料載入器
        criterion (nn.Module): 損失函數
        device (torch.device): 運算設備
        logger (Optional[logging.Logger]): 日誌記錄器

    Returns:
        Tuple[float, float, float, float]: (平均損失, 準確率, F1分數, 最佳閾值)
    """
    if logger is None:
        logger = logging.getLogger('StockModel')

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 收集所有預測和標籤
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            outputs = outputs.view(-1)
            labels = labels.float()

            # 確保輸出在 [0,1] 範圍內
            outputs = torch.clamp(outputs, 0, 1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 收集預測和標籤
            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 計算最佳閾值
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # 使用不同的閾值計算F1分數
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions = (all_outputs >= threshold).astype(float)
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # 使用最佳閾值計算最終指標
    predictions = (all_outputs >= best_threshold).astype(float)
    correct = np.sum(predictions == all_labels)
    total = len(all_labels)
    true_positives = np.sum((predictions == 1) & (all_labels == 1))
    false_positives = np.sum((predictions == 1) & (all_labels == 0))
    false_negatives = np.sum((predictions == 0) & (all_labels == 1))

    # 計算準確率
    accuracy = correct / total

    # 計算 F1 分數
    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return total_loss / len(data_loader), accuracy, f1, best_threshold


def main():
    """主函數"""
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='評估股票預測模型')
    parser.add_argument('--data_dir', type=str, default='split_purne',
                        help='資料目錄路徑')
    parser.add_argument('--output_dir', type=str, default='output/model_evaluation',
                        help='輸出目錄路徑')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='測試集比例')
    parser.add_argument('--random_state', type=int, default=42,
                        help='隨機種子')
    parser.add_argument('--model', type=str, required=True,
                        help='模型檔案路徑')
    parser.add_argument('--preprocessor', type=str, required=True,
                        help='預處理器檔案路徑')
    parser.add_argument('--test_mode', action='store_true',
                        help='是否使用測試模式（只使用小樣本）')
    parser.add_argument('--test_samples', type=int, default=5,
                        help='測試模式下的樣本數量')
    args = parser.parse_args()

    # 創建輸出目錄
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 設定日誌
    logger = setup_logger(output_dir)

    # 檢查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用設備: {device}")

    # 讀取資料
    logger.info("正在讀取資料...")
    df = read_csv_files(args.data_dir, test_mode=args.test_mode,
                        test_samples=args.test_samples)
    logger.info(f"讀取到 {len(df)} 筆資料")

    # 載入預處理器
    logger.info(f"正在載入預處理器：{args.preprocessor}")
    try:
        preprocessor = DataPreprocessor.load(args.preprocessor, logger)
    except Exception as e:
        logger.info(f"嘗試從目錄載入預處理器：{os.path.dirname(args.preprocessor)}")
        preprocessor = DataPreprocessor.load(
            os.path.dirname(args.preprocessor), logger)

    # 準備資料
    logger.info("正在準備資料...")
    X_train, X_test, y_train, y_test, _ = prepare_data(
        df, test_size=args.test_size, random_state=args.random_state,
        logger=logger, preprocessor=preprocessor
    )

    # 檢查類別分布
    train_pos_ratio = np.mean(y_train)
    test_pos_ratio = np.mean(y_test)
    logger.info(f"\n類別分布:")
    logger.info(f"訓練集正樣本比例: {train_pos_ratio:.4f}")
    logger.info(f"測試集正樣本比例: {test_pos_ratio:.4f}")

    # 創建資料集和資料載入器
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    # 創建模型
    input_size = X_train.shape[1]
    model = create_dnn_model(
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        dropout_rate=0.3,
        batch_norm=True,
        activation='leaky_relu'
    ).to(device)

    # 載入模型權重
    logger.info(f"正在載入模型：{args.model}")
    try:
        checkpoint = torch.load(args.model)
    except FileNotFoundError:
        # 嘗試不同的擴展名
        alt_model_path = args.model.replace('.pt', '.pth')
        if os.path.exists(alt_model_path):
            logger.info(f"找不到 {args.model}，嘗試載入 {alt_model_path}")
            checkpoint = torch.load(alt_model_path)
        else:
            raise FileNotFoundError(f"找不到模型文件：{args.model} 或 {alt_model_path}")

    # 檢查模型狀態字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_threshold = checkpoint.get('threshold', 0.5)
        logger.info(f"模型已載入，保存的閾值：{saved_threshold:.4f}")
    else:
        # 如果載入的是狀態字典
        model.load_state_dict(checkpoint)
        saved_threshold = 0.5
        logger.info("模型狀態字典已載入")

    # 定義損失函數
    pos_weight = (1 - train_pos_ratio) / train_pos_ratio
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device))

    # 評估模型
    logger.info("\n開始評估...")

    # 評估訓練集
    train_loss, train_acc, train_f1, train_threshold = evaluate(
        model, train_loader, criterion, device, logger)
    logger.info(f"\n訓練集結果:")
    logger.info(f"Loss: {train_loss:.4f}")
    logger.info(f"Accuracy: {train_acc:.4f}")
    logger.info(f"F1 Score: {train_f1:.4f}")
    logger.info(f"最佳閾值: {train_threshold:.4f}")

    # 評估測試集
    test_loss, test_acc, test_f1, test_threshold = evaluate(
        model, test_loader, criterion, device, logger)
    logger.info(f"\n測試集結果:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"Accuracy: {test_acc:.4f}")
    logger.info(f"F1 Score: {test_f1:.4f}")
    logger.info(f"最佳閾值: {test_threshold:.4f}")

    logger.info(f"\n評估完成！結果保存在: {output_dir}")


if __name__ == '__main__':
    main()
