import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import urllib.request # 新增：用于处理网络请求

# ==========================================
# 🌟 网络修复补丁：伪装成正常的浏览器，防止被服务器拒绝连接
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
urllib.request.install_opener(opener)
# ==========================================

# 定义进阶版 CNN 模型
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        # 卷积层 1 (1 -> 32 通道)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # 卷积层 2 (32 -> 64 通道)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        # 卷积层 3 (新增：64 -> 128 通道)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1) 
        )
        
        # 新增：Dropout 层，概率设为 0.5
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层：128通道 * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) 
        out = self.dropout(out) 
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用进阶网络结构进行训练，计算设备: {device}")

    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_size = 50000
    val_size = 10000
    train_data, val_data = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = AdvancedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练 5 个 epoch
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/5] 完成训练")

    # 测试最终准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"\n⭐ 进阶模型最终测试准确率: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()