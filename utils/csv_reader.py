import os
from typing import List, Optional
import pandas as pd
from tqdm import tqdm


def read_csv_files(
    path: str,
    test_mode: bool = False,
    test_samples: int = 1
) -> pd.DataFrame:
    """
    讀取指定路徑下的所有 CSV 檔案並合併成一個 DataFrame

    Args:
        path (str): CSV 檔案所在的目錄路徑
        test_mode (bool, optional): 是否為測試模式。預設為 False
        test_samples (int, optional): 測試模式下要讀取的檔案數量。預設為 1

    Returns:
        pd.DataFrame: 合併後的 DataFrame

    Raises:
        FileNotFoundError: 當指定的路徑不存在時
        ValueError: 當 test_samples 小於 1 時
    """
    # 檢查路徑是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"路徑 {path} 不存在")

    # 檢查 test_samples 是否有效
    if test_mode and test_samples < 1:
        raise ValueError("test_samples 必須大於 0")

    # 獲取所有 CSV 檔案
    csv_files: List[str] = [f for f in os.listdir(path) if f.endswith('.csv')]

    # 在測試模式下限制檔案數量
    if test_mode:
        csv_files = csv_files[:test_samples]

    # 使用 tqdm 顯示進度條
    dfs: List[pd.DataFrame] = []
    for file in tqdm(csv_files, desc="讀取 CSV 檔案"):
        file_path = os.path.join(path, file)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"讀取檔案 {file} 時發生錯誤: {str(e)}")
            continue

    # 合併所有 DataFrame
    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    # 測試程式碼
    try:
        # 測試正常模式
        df_normal = read_csv_files("split_purne")
        print(f"正常模式讀取到的資料筆數: {len(df_normal)}")

        # 測試模式
        df_test = read_csv_files("split_purne", test_mode=True, test_samples=2)
        print(f"測試模式讀取到的資料筆數: {len(df_test)}")

    except Exception as e:
        print(f"執行時發生錯誤: {str(e)}")
