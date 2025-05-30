import torch, os
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
BATCH_SIZE = 4096
EPOCHS = 100
DROPOUT_RATE = 0.5


# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, transform=transform),
                         batch_size=BATCH_SIZE)

class ImprovedCNN(nn.Module):
    def __init__(self, dropout:float=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# ---------------工具函数begin--------------
# 测试
def test(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            correct += (output.argmax(1) == y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    if(epoch <= EPOCHS):
        logging.info(f"Epoch {epoch+1}/{EPOCHS} CNN Accuracy: {accuracy:.4f}")
    else:
        logging.info(f"Final CNN Accuracy: {accuracy:.4f}")
    return accuracy


# 保存模型
def save_model(model, meta):
    os.makedirs('./models', exist_ok=True)
    model_save_path = f"./models/{meta}_{logger_name}_{iso_str}.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")


# ---------------工具函数end--------------
if __name__ == "__main__":

    # 设置日志
    logger_name = 'fashion_mnist_IPcnn'
    iso_str = datetime.now().isoformat().replace(':', '-')
    logging.basicConfig(
        filename=f'./logs/{logger_name}_{iso_str}.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    model = ImprovedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 登记使用的超参数
    logging.info(f"Batch Size: {BATCH_SIZE}")
    logging.info(f"Epochs: {EPOCHS}")
    logging.info(f"Dropout Rate: {DROPOUT_RATE}")
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    logging.info(f"Scheduler: {scheduler.__class__.__name__}")
    logging.info(f"Loss Function: {criterion.__class__.__name__}")
    logging.info(f"Device: {device}")
    logging.info(f"Training on {len(train_loader.dataset)} samples")
    logging.info(f"Testing on {len(test_loader.dataset)} samples")

    best_accuracy = 0.9299
    logging.info(f"Initial Best Accuracy: {best_accuracy:.4f}")
    
    for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch"):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        logging.info(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")
        acc = test(model, test_loader, epoch)
        if acc > best_accuracy:
            best_accuracy = acc
            save_model(model, f"epoch_{epoch+1}_accuracy_{best_accuracy:.4f}")
            logging.info(f"Best model saved with accuracy: {best_accuracy:.4f}")

    # 最终测试
    acc = test(model, test_loader, EPOCHS)
    if acc > best_accuracy:
        best_accuracy = acc
        save_model(model, f"final_accuracy_{best_accuracy:.4f}")
        logging.info(f"Final Best model saved with accuracy: {best_accuracy:.4f}")

    logging.info(f"Final CNN Accuracy: {best_accuracy:.4f}")