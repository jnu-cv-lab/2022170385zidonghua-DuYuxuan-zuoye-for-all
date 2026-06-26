import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import urllib.request

# 网络补丁：防止下载 MNIST 时连接被重置
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# 1. 重新声明我们的 CNN 模型结构 (任务 3 的内容)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print("="*50)
    print("Lab 09 任务 4：训练 CNN 模型")
    print("="*50)

    # 设备检测：有 GPU 就用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")

    # 2. 准备训练数据
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. 实例化模型、损失函数和优化器 [cite: 124-125]
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss() # 选择交叉熵损失函数，适用于多分类
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 选择 Adam 优化器，收敛快且稳定

    # 4. 开始训练循环 [cite: 126]
    epochs = 5
    print(f"\n🚀 开始执行 {epochs} 个 Epoch 的训练...")
    
    for epoch in range(epochs):
        model.train() # 切换到训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # 将数据搬运到 GPU/CPU
            images, labels = images.to(device), labels.to(device)
            
            # --- 核心训练四步走 ---
            optimizer.zero_grad()           # 1. 梯度清零
            outputs = model(images)         # 2. 前向传播（预测）
            loss = criterion(outputs, labels) # 3. 计算误差（Loss）
            loss.backward()                 # 4. 反向传播（求导）
            optimizer.step()                # 5. 更新参数（学习）
            
            # 统计数据 [cite: 127-128]
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # 计算该 Epoch 的平均 Loss 和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Training Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.4f}")

    print("\n✅ 任务 4 训练阶段顺利完成！")

if __name__ == "__main__":
    main()