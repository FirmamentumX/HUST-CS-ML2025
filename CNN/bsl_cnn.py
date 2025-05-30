import torch, os
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
from datetime import datetime

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
BATCH_SIZE = 128
EPOCHS = 10

# 设置日志
logger_name = 'fashion_mnist_cnn'
iso_str = datetime.now().isoformat().replace(':', '-')
logging.basicConfig(
    filename=f'./logs/{logger_name}_{iso_str}.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, transform=transform),
                         batch_size=BATCH_SIZE)

# 模型定义
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch"):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    logging.info(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")

# 测试
model.eval()
correct = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        correct += (output.argmax(1) == y).sum().item()
logging.info(f"CNN Accuracy: {correct/len(test_loader.dataset):.4f}")

# 保存模型
os.makedirs('./models', exist_ok=True)
model_save_path = f"./models/{logger_name}_{iso_str}.pth"
torch.save(model.state_dict(), model_save_path)
logging.info(f"Model saved to {model_save_path}")