import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import logging
import datetime, os
from utils import EnhancedWideResNet as ResNet, init_weights



# 超参数
BATCH_SIZE = 256
EPOCHS = 650
BEST_ACC = 0.959

# 数据加载
# train_transform = transforms.Compose([
#     transforms.RandomCrop(28, padding=4),
#     transforms.RandomHorizontalFlip(),
#     # transforms.RandomVerticalFlip(),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
#     transforms.ToTensor(),
#     transforms.Normalize((0.2860,), (0.3530,))
# ])
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2)]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
    transforms.RandomErasing(p=0.9, scale=(0.1, 0.2))  # CutOut
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
train_loader = DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=train_transform),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, transform=transform),
                         batch_size=BATCH_SIZE)

# 设置日志
logger_name = 'fashion_mnist_Iwrn_40-4-0.5'
iso_str = datetime.datetime.now().isoformat().replace(':', '-')
logging.basicConfig(
    filename=f'./logs/{logger_name}_{iso_str}.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
        
# 训练循环
def train(model, epochs, train_loader, criterion, optimizer, scheduler, device):
    global BEST_ACC
    best_score = BEST_ACC
    model.train()
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        acc = test(model, test_loader, epoch, device)        
        logging.info(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f} Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if acc > best_score:
            best_score = acc
            save_model(model, f"epoch_{epoch+1}_accuracy_{acc:.4f}")
            logging.info(f"Epoch {epoch+1}/{epochs} Model saved with accuracy: {acc:.4f}")
    BEST_ACC = best_score

# 测试
def test(model, test_loader, epoch, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            correct += (output.argmax(1) == y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    if(epoch <= EPOCHS):
        logging.info(f"Epoch {epoch+1}/{EPOCHS} RNN Accuracy: {accuracy:.4f}")
    else:
        logging_end(accuracy)
    return accuracy

# 保存模型
def save_model(model, meta):
    os.makedirs('./models', exist_ok=True)
    model_save_path = f"./models/{meta}_{logger_name}_{iso_str}.pth"
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

# 微调模型
def setup_finetune_layers(model, layers_to_finetune):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_finetune):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
    
def setup_finetune(model, layers):
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 解冻顶层
    for param in model.net[-layers:].parameters():  # 微调最后几层
        param.requires_grad = True

    return model

def logging_start(device, logger_name, model, optimizer, scheduler=None, criterion=nn.CrossEntropyLoss()):
    logging.info(f"Training started at {datetime.datetime.now().isoformat()}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Epochs: {EPOCHS}")
    logging.info(f"Device: {device}")
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Optimizer: {optimizer.__class__.__name__}")
    logging.info(f"Scheduler: {scheduler.__class__.__name__}")
    logging.info(f"Loss Function: {criterion.__class__.__name__}")
    logging.info(f"Device: {device}")
    logging.info(f"Training on {len(train_loader.dataset)} samples")
    logging.info(f"Testing on {len(test_loader.dataset)} samples")
    # logging.info(f"Transform: {train_transform}")
    logging.info(f"Original Best accuracy: {BEST_ACC:.4f}")
    logging.info(f"Training started at {datetime.datetime.now().isoformat()}")

def logging_end(accuracy):
    logging.info(f"Training finished at {datetime.datetime.now().isoformat()}")
    logging.info(f"Best RNN Accuracy: {BEST_ACC:.4f}")

# 加载模型
def loadin_model(path, device, **kwargs):
    model = ResNet(**kwargs)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

def main():

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 训练
    # model = loadin_model('./models/epoch_544_accuracy_0.9597_fashion_mnist_Iwrn_40-4-0.5_2025-05-28T17-22-08.164804.pth', 
    #                     device,
    #                     depth=40, 
    #                     widen_factor=4, 
    #                     dropout_rate=0.5)
    # model = setup_finetune_layers(model, layers_to_finetune = ['fc']) 

    # model = ResNet().to(device)
    model = ResNet(depth=40, widen_factor=4, dropout_rate=0.5).to(device) 
    init_weights(model, 'kaiming')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=3e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.002, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3
    )
    logging_start(device, logger_name, model, optimizer, scheduler=scheduler, criterion=criterion)
    train(model, epochs=EPOCHS, train_loader=train_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device)
    test(model, test_loader, EPOCHS+1, device=device)

if __name__ == "__main__":
    main()