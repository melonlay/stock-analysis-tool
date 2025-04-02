import os
import matplotlib.pyplot as plt


def plot_training_history(
    train_losses: list,
    train_accuracies: list,
    train_f1s: list,
    val_losses: list,
    val_accuracies: list,
    val_f1s: list,
    save_dir: str
) -> None:
    """
    繪製訓練歷史圖表

    Args:
        train_losses (list): 訓練損失歷史
        train_accuracies (list): 訓練準確率歷史
        train_f1s (list): 訓練F1分數歷史
        val_losses (list): 驗證損失歷史
        val_accuracies (list): 驗證準確率歷史
        val_f1s (list): 驗證F1分數歷史
        save_dir (str): 圖表保存目錄
    """
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
    plt.rcParams['axes.unicode_minus'] = False  # 用來正常顯示負號

    plt.figure(figsize=(15, 5))

    # 損失圖表
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='訓練損失')
    plt.plot(val_losses, label='驗證損失')
    plt.title('損失歷史')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.legend()

    # 準確率圖表
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='訓練準確率')
    plt.plot(val_accuracies, label='驗證準確率')
    plt.title('準確率歷史')
    plt.xlabel('Epoch')
    plt.ylabel('準確率')
    plt.legend()

    # F1分數圖表
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='訓練F1')
    plt.plot(val_f1s, label='驗證F1')
    plt.title('F1分數歷史')
    plt.xlabel('Epoch')
    plt.ylabel('F1分數')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
