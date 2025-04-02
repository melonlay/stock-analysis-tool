"""
模型訓練主程式
"""
import os
import sys
import argparse
import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score

# 添加專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..utils.csv_reader import read_csv_files
    from ..utils.data_preprocessor import DataPreprocessor
    from ..utils.prepare_data import prepare_data
    from ..utils.data_balancer import balance_data
    from ..model.dnn import create_dnn_model
    from ..utils.plot_utils import plot_training_history
    from ..utils.data_augmentor import DataAugmentor
    from .dataset.StockDataset import StockDataset
    from .evaluator import evaluate
    from ..config.model_config import (
        TRAINING_CONFIG,
        AUGMENTATION_CONFIG,
        PREPROCESSING_CONFIG,
        VALIDATION_CONFIG,
        LOGGING_CONFIG,
        SAVE_CONFIG
    )
except ImportError:
    from utils.csv_reader import read_csv_files
    from utils.data_preprocessor import DataPreprocessor
    from utils.prepare_data import prepare_data
    from utils.data_balancer import balance_data
    from model.dnn import create_dnn_model
    from utils.plot_utils import plot_training_history
    from utils.data_augmentor import DataAugmentor
    from trainer.dataset.StockDataset import StockDataset
    from trainer.evaluator import evaluate
    from config.model_config import (
        TRAINING_CONFIG,
        AUGMENTATION_CONFIG,
        PREPROCESSING_CONFIG,
        VALIDATION_CONFIG,
        LOGGING_CONFIG,
        SAVE_CONFIG
    )


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
    log_file = os.path.join(log_dir, 'training.log')
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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger: Optional[logging.Logger],
    augmentor: Optional[DataAugmentor] = None,
    positive_only: bool = True
) -> Tuple[float, float, float, float]:
    """
    訓練一個 epoch

    Args:
        model (nn.Module): 模型
        train_loader (DataLoader): 訓練資料載入器
        criterion (nn.Module): 損失函數
        optimizer (optim.Optimizer): 優化器
        device (torch.device): 運算設備
        logger (Optional[logging.Logger]): 日誌記錄器
        augmentor (Optional[DataAugmentor]): 資料增強器
        positive_only (bool): 是否只對正樣本進行增強

    Returns:
        Tuple[float, float, float, float]: (平均損失, 準確率, F1分數, 最佳閾值)
    """
    if logger is None:
        logger = logging.getLogger('StockModel')

    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_outputs = []
    all_labels = []

    for batch_idx, (X, y) in enumerate(train_loader):
        # 將資料移到設備上
        X = X.to(device)
        y = y.to(device)

        # 在正向傳遞前進行資料增強
        if augmentor is not None:
            X, y = augmentor.augment_batch(X, y, positive_only=positive_only)

        # 前向傳遞
        outputs = model(X)
        loss = criterion(outputs, y.unsqueeze(1))

        # 反向傳遞和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 收集預測結果
        probs = torch.sigmoid(outputs)
        all_outputs.extend(probs.detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        # 更新統計資訊
        total_loss += loss.item()
        predictions = (probs > 0.5).float()
        total_correct += (predictions.squeeze() == y).sum().item()
        total_samples += len(y)

    # 計算最終指標
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    # 尋找最佳閾值和計算 F1 分數
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions = (all_outputs > threshold).astype(int)
        f1 = f1_score(all_labels, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return avg_loss, accuracy, best_f1, best_threshold


def prepare_data(
    df: pd.DataFrame,
    use_augmentation: bool = False,
    logger: Optional[logging.Logger] = None,
    preprocessor: Optional[DataPreprocessor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DataPreprocessor, Optional[DataAugmentor]]:
    """
    準備訓練資料

    Args:
        df: 已讀取的資料框
        use_augmentation: 是否使用資料增強
        logger: 日誌記錄器
        preprocessor: 預處理器，如果提供則使用現有的

    Returns:
        訓練集特徵、測試集特徵、訓練集標籤、測試集標籤、預處理器、資料增強器
    """
    if logger is None:
        logger = logging.getLogger('StockModel')

    label_column = PREPROCESSING_CONFIG['label_column']
    logger.info(f"原始資料集大小: {len(df)}")
    logger.info(f"原始正樣本比例: {df[label_column].mean():.4f}")

    # 資料預處理
    logger.info("正在進行資料預處理...")
    if preprocessor is None:
        preprocessor = DataPreprocessor(
            standardize=PREPROCESSING_CONFIG['standardize'],
            handle_missing=PREPROCESSING_CONFIG['handle_missing'],
            detect_outliers=PREPROCESSING_CONFIG['outlier_detection'],
            feature_selection=PREPROCESSING_CONFIG['feature_selection'],
            logger=logger
        )
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)

    # 分割資料
    logger.info("正在分割資料...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=VALIDATION_CONFIG['test_size'],
        random_state=VALIDATION_CONFIG['random_state'],
        stratify=y if VALIDATION_CONFIG['stratify'] else None
    )

    # 初始化資料增強器（如果需要）
    augmentor = None
    if use_augmentation:
        logger.info("正在初始化資料增強器...")
        augmentor = DataAugmentor(logger=logger)

        # 分析資料特性並調整噪聲參數
        stats = augmentor.analyze_data_characteristics(X_train)
        augmentor.adjust_noise_levels(stats)

        # 記錄資料特性分析結果
        logger.info("\n資料特性分析結果:")
        mean_stats = {
            'std': np.mean([feature_stats['std'] for feature_stats in stats.values()]),
            'skewness': np.mean([feature_stats['skewness'] for feature_stats in stats.values()]),
            'kurtosis': np.mean([feature_stats['kurtosis'] for feature_stats in stats.values()]),
            'outlier_ratio': np.mean([feature_stats['outlier_ratio'] for feature_stats in stats.values()]),
            'trend': np.mean([feature_stats['trend'] for feature_stats in stats.values()]),
            'seasonality': np.mean([feature_stats['seasonality'] for feature_stats in stats.values()])
        }

        logger.info(f"平均標準差: {mean_stats['std']:.4f}")
        logger.info(f"平均偏度: {mean_stats['skewness']:.4f}")
        logger.info(f"平均峰度: {mean_stats['kurtosis']:.4f}")
        logger.info(f"平均異常值比例: {mean_stats['outlier_ratio']:.4f}")
        logger.info(f"平均趨勢強度: {mean_stats['trend']:.4f}")
        logger.info(f"平均季節性強度: {mean_stats['seasonality']:.4f}")

    return X_train, X_test, y_train, y_test, preprocessor, augmentor


def main():
    """主函數"""
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='訓練股票預測模型')
    parser.add_argument('--data_dir', type=str, default='split_purne',
                        help='資料目錄路徑')
    parser.add_argument('--output_dir', type=str, default='output/model_training',
                        help='輸出目錄路徑')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=TRAINING_CONFIG['learning_rate'],
                        help='學習率')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                        help='訓練輪數')
    parser.add_argument('--test_size', type=float, default=VALIDATION_CONFIG['test_size'],
                        help='測試集比例')
    parser.add_argument('--random_state', type=int, default=VALIDATION_CONFIG['random_state'],
                        help='隨機種子')
    parser.add_argument('--test_mode', action='store_true',
                        help='是否使用測試模式')
    parser.add_argument('--test_samples', type=int, default=5,
                        help='測試模式下要讀取的檔案數量')
    parser.add_argument('--save_preprocessor', action='store_true',
                        help='是否儲存預處理器狀態')
    parser.add_argument('--preprocessor', type=str, default=None,
                        help='預處理器檔案路徑，如果提供則使用現有的預處理器')
    parser.add_argument('--use_augmentation', action='store_true',
                        default=TRAINING_CONFIG['use_augmentation'],
                        help='是否使用資料增強')
    parser.add_argument('--n_augment', type=int, default=TRAINING_CONFIG['n_augment'],
                        help='每個樣本要增強幾次')
    parser.add_argument('--augment_all', action='store_true',
                        default=not TRAINING_CONFIG['positive_only_augment'],
                        help='是否對所有樣本進行增強（預設只對正樣本）')
    parser.add_argument('--patience', type=int, default=TRAINING_CONFIG['patience'],
                        help='早停的耐心值（多少個 epoch 沒有改善就停止）')
    parser.add_argument('--hidden_sizes', type=str, default=','.join(map(str, TRAINING_CONFIG['hidden_sizes'])),
                        help='隱藏層大小，以逗號分隔')
    parser.add_argument('--dropout_rate', type=float, default=TRAINING_CONFIG['dropout_rate'],
                        help='Dropout率')
    parser.add_argument('--weight_decay', type=float, default=TRAINING_CONFIG['weight_decay'],
                        help='權重衰減率')
    parser.add_argument('--pos_weight_multiplier', type=float, default=TRAINING_CONFIG['pos_weight_multiplier'],
                        help='正樣本權重倍數')
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
    df = read_csv_files(
        args.data_dir,
        test_mode=args.test_mode,
        test_samples=args.test_samples
    )
    logger.info(f"讀取到 {len(df)} 筆資料")

    # 準備資料
    logger.info("正在準備資料...")

    # 如果有提供預處理器檔案，則載入它
    preprocessor = None
    if args.preprocessor:
        logger.info(f"正在載入預處理器：{args.preprocessor}")
        preprocessor = DataPreprocessor.load(args.preprocessor, logger)

    X_train, X_test, y_train, y_test, preprocessor, augmentor = prepare_data(
        df, args.use_augmentation, logger, preprocessor)

    # 檢查類別分布
    train_pos_ratio = np.mean(y_train)
    test_pos_ratio = np.mean(y_test)
    logger.info(f"\n類別分布:")
    logger.info(f"訓練集正樣本比例: {train_pos_ratio:.4f}")
    logger.info(f"測試集正樣本比例: {test_pos_ratio:.4f}")

    # 如果要求儲存預處理器，則儲存它
    if args.save_preprocessor:
        preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
        logger.info(f"正在儲存預處理器到：{preprocessor_path}")
        preprocessor.save(preprocessor_path)

    # 創建資料集和資料載入器
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    # 創建模型
    input_size = X_train.shape[1]
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
    model = create_dnn_model(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=args.dropout_rate,
        batch_norm=True,
        activation='relu'
    ).to(device)

    # 輸出模型架構
    logger.info("\n模型架構:")
    logger.info(f"""
DNN 模型摘要:
輸入維度: {input_size}
隱藏層: {hidden_sizes}
Dropout率: {args.dropout_rate}
批次正規化: 是
激活函數: leaky_relu
總參數數量: {sum(p.numel() for p in model.parameters())}
可訓練參數數量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}
""")

    # 定義損失函數
    pos_weight = (1 - train_pos_ratio) / train_pos_ratio * \
        args.pos_weight_multiplier
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device))

    # 定義優化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 定義學習率排程器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=TRAINING_CONFIG['lr_scheduler']['factor'],
        patience=TRAINING_CONFIG['lr_scheduler']['patience'],
        min_lr=TRAINING_CONFIG['lr_scheduler']['min_lr'],
        verbose=True
    )

    # 訓練模型
    logger.info("\n開始訓練...")
    best_val_f1 = 0.0
    best_model_path = None
    best_threshold = 0.5
    patience_counter = 0

    train_losses = []
    train_accuracies = []
    train_f1s = []
    train_thresholds = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    val_thresholds = []

    for epoch in tqdm(range(args.epochs), desc="訓練進度"):
        # 訓練一個 epoch
        train_loss, train_acc, train_f1, train_threshold = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, augmentor)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        train_thresholds.append(train_threshold)

        # 驗證
        val_loss, val_acc, val_f1, val_threshold = evaluate(
            model, test_loader, criterion, device, logger)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        val_thresholds.append(val_threshold)

        # 更新學習率
        scheduler.step(val_f1)

        # 記錄訓練結果
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Threshold: {train_threshold:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Threshold: {val_threshold:.4f}"
        )

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = val_threshold
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold,
                'config': {
                    'input_size': input_size,
                    'hidden_sizes': hidden_sizes,
                    'dropout_rate': args.dropout_rate,
                    'batch_norm': True,
                    'activation': 'relu'
                }
            }, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= args.patience:
            logger.info(f"早停：{args.patience} 個 epoch 沒有改善")
            break

        # 定期保存檢查點
        if SAVE_CONFIG['save_checkpoints'] and (epoch + 1) % SAVE_CONFIG['checkpoint_frequency'] == 0:
            checkpoint_path = os.path.join(
                output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'best_threshold': best_threshold
            }, checkpoint_path)

    # 保存最後一個模型
    if SAVE_CONFIG['save_last_model']:
        last_model_path = os.path.join(output_dir, 'last_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'threshold': val_threshold,
            'config': {
                'input_size': input_size,
                'hidden_sizes': hidden_sizes,
                'dropout_rate': args.dropout_rate,
                'batch_norm': True,
                'activation': 'relu'
            }
        }, last_model_path)

    # 繪製訓練歷史
    logger.info("\n繪製訓練歷史圖表...")
    plot_training_history(
        train_losses, train_accuracies, train_f1s,
        val_losses, val_accuracies, val_f1s,
        output_dir
    )

    logger.info(f"\n訓練完成！結果保存在: {output_dir}")
    logger.info(f"最佳驗證F1: {best_val_f1:.4f}")
    logger.info(f"最佳閾值: {best_threshold:.4f}")


if __name__ == '__main__':
    main()
