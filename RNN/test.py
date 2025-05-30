import torch,os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from .utils import EnhancedWideResNet as ResNet

BATCH_SIZE = 256


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, transform=transform),
                         batch_size=BATCH_SIZE)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(depth=40, widen_factor=4, dropout_rate=0.5).to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 加载预训练模型
    model_path = f'{current_dir}/models/epoch_565_accuracy_0.9609_fashion_mnist_Iwrn_40-4-0.5_2025-05-29T23-37-24.595442.pth'
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    
    # 测试模型
    test(model, test_loader, device)

if __name__ == "__main__":
    main()