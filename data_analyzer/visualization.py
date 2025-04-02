"""
特徵重要性視覺化模組
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import os
from .base import ImportanceMethod

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 用來正常顯示負號


def plot_feature_importance(
    importance: pd.DataFrame,
    method: ImportanceMethod,
    output_dir: str = 'output/data_analysis',
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    繪製特徵重要性圖表

    Args:
        importance (pd.DataFrame): 特徵重要性數據框
        method (ImportanceMethod): 使用的分析方法
        output_dir (str, optional): 輸出目錄路徑。預設為 'output/data_analysis'
        top_n (int, optional): 要顯示的前 N 個特徵。預設為 20
        figsize (Tuple[int, int], optional): 圖表大小。預設為 (12, 8)
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    sns.barplot(
        data=importance.head(top_n),
        x='importance',
        y='feature'
    )

    method_names = {
        ImportanceMethod.RANDOM_FOREST: '隨機森林',
        ImportanceMethod.XGBOOST: 'XGBoost',
        ImportanceMethod.LIGHTGBM: 'LightGBM',
        ImportanceMethod.MUTUAL_INFO: '互信息'
    }

    plt.title(f'{method_names[method]} - 前 {top_n} 個重要特徵', fontsize=14)
    plt.xlabel('重要性', fontsize=12)
    plt.ylabel('特徵', fontsize=12)
    plt.tight_layout()

    # 使用新的輸出路徑
    output_path = os.path.join(output_dir, f'feature_importance_{method}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
