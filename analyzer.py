"""
特徵重要性分析主程式
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
import os
from utils.csv_reader import read_csv_files
from data_analyzer import (
    ImportanceMethod,
    RandomForestAnalyzer,
    XGBoostAnalyzer,
    LightGBMAnalyzer,
    MutualInfoAnalyzer,
    plot_feature_importance,
    preprocess_features
)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    準備數據，將特徵和目標分離，並進行特徵預處理

    Args:
        df (pd.DataFrame): 原始數據框

    Returns:
        tuple[pd.DataFrame, np.ndarray]: 特徵數據框和目標序列
    """
    # 移除 '飆股' 欄位作為目標變數
    y = df['飆股']

    # 移除 ID 欄位和目標變數
    X = df.drop(['飆股', 'ID'], axis=1, errors='ignore')

    # 對目標變數進行編碼
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 對特徵進行預處理
    X = preprocess_features(X)

    return X, y


def get_analyzer(method: ImportanceMethod, n_estimators: int):
    """
    根據指定的方法獲取對應的分析器

    Args:
        method (ImportanceMethod): 分析方法
        n_estimators (int): 決策樹數量

    Returns:
        BaseAnalyzer: 特徵重要性分析器
    """
    analyzers = {
        ImportanceMethod.RANDOM_FOREST: RandomForestAnalyzer,
        ImportanceMethod.XGBOOST: XGBoostAnalyzer,
        ImportanceMethod.LIGHTGBM: LightGBMAnalyzer,
        ImportanceMethod.MUTUAL_INFO: MutualInfoAnalyzer
    }

    return analyzers[method](n_estimators=n_estimators)


def analyze_with_method(X: pd.DataFrame, y: np.ndarray, method: ImportanceMethod, n_estimators: int, output_dir: str) -> None:
    """
    使用指定方法進行特徵重要性分析

    Args:
        X (pd.DataFrame): 特徵數據
        y (np.ndarray): 目標變數
        method (ImportanceMethod): 分析方法
        n_estimators (int): 決策樹數量
        output_dir (str): 輸出目錄路徑
    """
    print(f"\n正在使用 {method} 方法分析特徵重要性...")
    analyzer = get_analyzer(method, n_estimators)
    importance = analyzer.analyze(X, y)

    # 輸出前 10 個重要特徵
    print(f"\n使用 {method} 方法的前 10 個重要特徵：")
    print(importance.head(10))

    # 繪製圖表
    print("\n正在繪製特徵重要性圖表...")
    plot_feature_importance(importance, method, output_dir=output_dir)
    print(
        f"圖表已保存為 '{os.path.join(output_dir, f'feature_importance_{method}.png')}'")

    # 保存所有特徵的結果到 CSV
    csv_path = os.path.join(output_dir, f'feature_importance_{method}.csv')
    importance.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"完整特徵重要性結果已保存為 '{csv_path}'")


def main():
    """主函數：讀取數據並進行特徵重要性分析"""
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='分析特徵重要性')
    parser.add_argument('--test_mode', action='store_true',
                        help='是否使用測試模式（只讀取部分數據）')
    parser.add_argument('--test_samples', type=int,
                        default=2, help='測試模式下要讀取的檔案數量（預設：2）')
    parser.add_argument('--method', type=ImportanceMethod, choices=list(ImportanceMethod),
                        default=ImportanceMethod.RANDOM_FOREST,
                        help='特徵重要性分析方法 (rf: 隨機森林, xgb: XGBoost, lgb: LightGBM, mi: 互信息)')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='樹模型的決策樹數量（預設：100）')
    parser.add_argument('--test_all', action='store_true',
                        help='測試所有分析方法')
    parser.add_argument('--input_dir', type=str, default='split_purne',
                        help='輸入資料夾路徑（預設：split_purne）')
    parser.add_argument('--output_dir', type=str, default='output/data_analysis',
                        help='輸出資料夾路徑（預設：output/data_analysis）')
    args = parser.parse_args()

    # 確保輸出目錄存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 讀取數據
    print("正在讀取數據...")
    df = read_csv_files(args.input_dir, test_mode=args.test_mode,
                        test_samples=args.test_samples)

    # 準備數據
    print("正在準備數據...")
    X, y = prepare_data(df)

    if args.test_all:
        # 測試所有方法
        for method in ImportanceMethod:
            analyze_with_method(
                X, y, method, args.n_estimators, args.output_dir)
            print("\n" + "="*50 + "\n")  # 分隔線
    else:
        # 使用單一方法
        analyze_with_method(X, y, args.method,
                            args.n_estimators, args.output_dir)


if __name__ == "__main__":
    main()
