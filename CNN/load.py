from .do_cnn import test_loader
import torch,os
import torch.nn as nn
from sklearn.metrics import f1_score
# 模型定义
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 256), nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.net(x)

def loadin_docnn(path, device):
    model = CNN()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

# 测试
def test(model, test_loader, device):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            preds = output.argmax(dim=1)

            correct += (preds == y).sum().item()

            # 收集预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)

    # 计算 F1 分数
    f1 = f1_score(all_targets, all_preds, average='macro')

    print(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    return accuracy, f1


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = loadin_docnn(f'{current_dir}/models/epoch_99_accuracy_0.9335_fashion_mnist_Icnn_2025-05-25T20-24-29.353102.pth', device)
    acc = test(model, test_loader, device)

if __name__ == "__main__":
    main()