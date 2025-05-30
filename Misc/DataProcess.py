import torch
from torchvision import datasets, transforms

# 设置数据转换为 Tensor，并归一化到 [0, 1]
transform = transforms.ToTensor()

# 加载训练集
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 获取所有标签并统计类别分布
labels = train_dataset.targets
class_counts = labels.bincount()
print("类别分布：", class_counts.tolist())

# 判断是否类别均衡（每个类别是否都有6000个样本）
is_balanced = all(count == 6000 for count in class_counts)
print("是否类别均衡：", is_balanced)

# 加载数据并计算全局均值和标准差
loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000)

total_pixels = 0
sum_pixels = 0.0
sum_sq_pixels = 0.0

for images, _ in loader:
    # 展平为一维向量，形状为 [batch_size * 1 * 28 * 28]
    batch_pixels = images.view(-1)
    total_pixels += batch_pixels.numel()
    sum_pixels += batch_pixels.sum().item()
    sum_sq_pixels += (batch_pixels ** 2).sum().item()

# 计算均值和标准差
mean = sum_pixels / total_pixels
std = (sum_sq_pixels / total_pixels - mean ** 2) ** 0.5

print(f"数据集均值：{mean:.4f}")
print(f"数据集标准差：{std:.4f}")