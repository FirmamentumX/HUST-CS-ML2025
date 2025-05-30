import re
import matplotlib.pyplot as plt






# 解析日志文件
def parse_cnn_log(file_path):
    epochs = []
    losses = []
    accuracies = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取loss
            loss_match = re.search(r'Epoch (\d+)/\d+ Loss: (\d+\.\d+)', line)
            if loss_match:
                epoch = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                epochs.append(epoch)
                losses.append(loss)
                
            # 提取accuracy
            acc_match = re.search(r'Epoch (\d+)/\d+ CNN Accuracy: (\d+\.\d+)', line)
            if acc_match:
                accuracy = float(acc_match.group(2))
                accuracies.append(accuracy)
                
        # 确保三个列表长度一致（以 epochs 为准）
    min_len = min(len(epochs), len(losses), len(accuracies))
    return epochs[:min_len], losses[:min_len], accuracies[:min_len], None

def parse_rnn_log(file_path):
    epochs = []
    losses = []
    accuracies = []
    learning_rates = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取 loss 和 learning rate
            loss_match = re.search(r'Epoch (\d+)/\d+ Loss: (\d+\.\d+) Learning Rate: (\d+\.\d+)', line)
            if loss_match:
                epoch = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                lr = float(loss_match.group(3))
                epochs.append(epoch)
                losses.append(loss)
                learning_rates.append(lr)
                    

            # 提取 accuracy
            acc_match = re.search(r'Epoch (\d+)/\d+ RNN Accuracy: (\d+\.\d+)', line)
            if acc_match:
                epoch = int(acc_match.group(1))
                accuracy = float(acc_match.group(2))
                accuracies.append(accuracy)

    # 保证三个列表长度一致（以 epochs 为准）
    min_len = min(len(epochs), len(losses), len(accuracies))
    return (
        epochs[:min_len],
        losses[:min_len],
        accuracies[:min_len],
        learning_rates[:min_len]
    )


# 可视化函数
def plot_metrics(model_type, epochs, losses, accuracies, learning_rates=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Loss 子图
    ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=4)
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(min(losses) * 0.9, max(losses) * 1.1)

    # Accuracy 子图
    ax2.plot(epochs, accuracies, 'r-o', linewidth=2, markersize=4)
    lab = '[pretrained]' if model_type.lower() == 'cnn' else ''
    ax2.set_title(f'{model_type.upper()} Accuracy vs. Epochs {lab}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(accuracies) * 0.95, 1.0)

    plt.tight_layout()
    plt.savefig(f'training_metrics_{model_type.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 如果有学习率，单独绘图
    if learning_rates:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, learning_rates, 'g-o', linewidth=1.5, markersize=3)
        plt.title('Learning Rate vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'learning_rate_{model_type.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()

# 绘图类型转换
draw_dict = {
    'cnn': parse_cnn_log,
    'rnn': parse_rnn_log
}

# 绘图
def draw(log_file, model_type):
    if model_type.lower() not in draw_dict:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(draw_dict.keys())}")
    
    epochs, losses, accuracies, learning_rates = draw_dict[model_type.lower()](log_file)
    plot_metrics(model_type, epochs, losses, accuracies, learning_rates)
