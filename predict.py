import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import argparse

# 添加專案根目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.data_preprocessor import DataPreprocessor
    from trainer.dataset.StockDataset import StockDataset
    from model.dnn import create_dnn_model, DNN
except ImportError:
    from utils.data_preprocessor import DataPreprocessor
    from trainer.dataset.StockDataset import StockDataset
    from model.dnn import create_dnn_model, DNN


def load_model(model_path, device, input_size):
    """載入訓練好的模型"""
    try:
        # 嘗試直接載入狀態字典
        state_dict = torch.load(model_path, map_location=device)

        # 創建新模型（使用與訓練時相同的配置）
        model = create_dnn_model(
            input_size=input_size,
            hidden_sizes=[512, 256, 128],
            dropout_rate=0.3,
            batch_norm=True,
            activation='leaky_relu'
        )

        # 如果載入的是狀態字典格式
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"找不到模型檔案 {model_path}")
        return None
    except Exception as e:
        print(f"載入模型時發生錯誤: {str(e)}")
        return None


def predict_stocks(model_path, preprocessor_path, test_data_path, template_path, output_path, test_mode=False, test_sample=None, threshold=0.5):
    """預測飆股並生成提交檔案"""
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    print(f"預測閾值: {threshold}")

    # 讀取測試數據
    test_data = pd.read_csv(test_data_path)

    # 如果是測試模式，只取指定數量的樣本
    if test_mode and test_sample is not None:
        test_data = test_data.head(test_sample)
        print(f"測試模式：只使用前 {test_sample} 筆資料")

    # 使用預處理器轉換數據
    processed_data = DataPreprocessor.transform_unlabeled(
        test_data, preprocessor_path)

    # 獲取特徵維度
    input_size = processed_data.shape[1]
    print(f"特徵維度: {input_size}")

    # 載入模型
    model = load_model(model_path, device, input_size)
    if model is None:
        return

    # 創建數據集和數據載入器
    dummy_labels = np.zeros(len(processed_data))  # 創建全零的標籤陣列
    test_dataset = StockDataset(processed_data, dummy_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 進行預測
    predictions = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            outputs = model(features)
            outputs = outputs.view(-1)
            predictions.extend(outputs.cpu().numpy())

    # 將預測結果轉換為二分類（0或1）
    predictions = np.array(predictions)
    predictions = (predictions >= threshold).astype(int)

    # 讀取提交模板
    submission_template = pd.read_csv(template_path)

    # 如果是測試模式，只更新對應數量的預測結果
    if test_mode and test_sample is not None:
        submission_template = submission_template.head(test_sample)

    # 更新預測結果
    submission_template['飆股'] = predictions

    # 儲存結果
    submission_template.to_csv(output_path, index=False)
    print(f"預測完成，結果已儲存至 {output_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='股票飆股預測程式')
    parser.add_argument('--model', type=str, default='output/model_training/best_model.pth',
                        help='模型檔案路徑')
    parser.add_argument('--preprocessor', type=str, default='output/model_training/preprocessor_state.joblib',
                        help='預處理器狀態檔案路徑')
    parser.add_argument('--test_data', type=str, default='test_data/prune_public.csv',
                        help='測試數據檔案路徑')
    parser.add_argument('--template', type=str, default='test_data/submission_template_public.csv',
                        help='提交模板檔案路徑')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='輸出檔案路徑')
    parser.add_argument('--test_mode', action='store_true',
                        help='啟用測試模式')
    parser.add_argument('--test_sample', type=int, default=None,
                        help='測試模式下使用的樣本數量')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='預測閾值，預設為 0.5')

    args = parser.parse_args()

    predict_stocks(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        test_data_path=args.test_data,
        template_path=args.template,
        output_path=args.output,
        test_mode=args.test_mode,
        test_sample=args.test_sample,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
