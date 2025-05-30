import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return torch.relu(x)

# 残差网络
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

# 添加Dropout的残差网络
class ResNet_D(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)



class IResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(IResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        return F.relu(out)

# WRN-40-4 模型
class WideResNet(nn.Module):
    def __init__(self, depth=40, widen_factor=4, num_classes=10, dropout_rate=0.3):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'Depth should be 6n + 4'
        n = (depth - 4) // 6  # 每个 stage 的 block 数量为 6

        # 初始卷积层
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # 三阶段构建，每个阶段通道数都更多（变宽）
        self._make_stage(16, 16 * widen_factor, n, stride=1, dropout_rate=dropout_rate)
        self._make_stage(16 * widen_factor, 32 * widen_factor, n, stride=2, dropout_rate=dropout_rate)
        self._make_stage(32 * widen_factor, 64 * widen_factor, n, stride=2, dropout_rate=dropout_rate)

        # 添加最终分类层
        self.net.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.net.append(nn.Flatten())
        self.net.append(nn.Linear(64 * widen_factor, num_classes))

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        self.net.append(IResidualBlock(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        for _ in range(1, num_blocks):
            self.net.append(IResidualBlock(out_channels, out_channels, dropout_rate=dropout_rate))

    def forward(self, x):
        return self.net(x)



def init_weights(model, init_type='kaiming'):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'kaiming':
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight)
            else:
                raise ValueError(f"Unsupported init_type: {init_type}")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# 1. SE Block（通道注意力）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 2. 预激活 + SE 的残差块（PreAct + SE）
class SEPreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(SEPreActBlock, self).__init__()
        # print(f"SEPreActBlock: in_channels={in_channels}, out_channels={out_channels}")
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.se = SEBlock(out_channels)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        out += self.shortcut(x)
        return out

# 3. Stochastic Depth 包裹器（仅训练时生效）
class StochasticBlock(nn.Module):
    def __init__(self, block, survival_prob=0.8):
        super(StochasticBlock, self).__init__()
        self.block = block
        self.survival_prob = survival_prob

    def forward(self, x):
        if self.training and torch.rand(1).item() > self.survival_prob:
            return x  # 跳过该块
        return self.block(x)

# 4. 最终增强型 WideResNet 模型
class EnhancedWideResNet(nn.Module):
    def __init__(self, depth=64, widen_factor=8, num_classes=10, dropout_rate=0.3, survival_prob=0.8):
        super(EnhancedWideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'Depth should be 6n + 4'
        n = (depth - 4) // 6  # 每个 stage 的 block 数量

        # 初始卷积层
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # 三阶段构建（使用 SEPreActBlock + Stochastic Depth）
        self._make_stage(16, 16 * widen_factor, n, stride=1, dropout_rate=dropout_rate, survival_prob=survival_prob)
        self._make_stage(16 * widen_factor, 32 * widen_factor, n, stride=2, dropout_rate=dropout_rate, survival_prob=survival_prob)
        self._make_stage(32 * widen_factor, 64 * widen_factor, n, stride=2, dropout_rate=dropout_rate, survival_prob=survival_prob)

        # 分类头
        self.net.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.net.append(nn.Flatten())
        self.net.append(nn.Linear(64 * widen_factor, num_classes))

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, dropout_rate, survival_prob):
        # 第一个 block 有 stride
        block = SEPreActBlock(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate)
        self.net.append(block) # 这个块要改变通道数，不能用StochasticBlock跳过，否则爆炸
        # 后续 blocks 无 stride
        for _ in range(1, num_blocks):
            block = SEPreActBlock(out_channels, out_channels, dropout_rate=dropout_rate)
            self.net.append(StochasticBlock(block, survival_prob=survival_prob))

    def forward(self, x):
        return self.net(x)