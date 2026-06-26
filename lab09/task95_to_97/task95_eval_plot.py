import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import urllib.request

# 网络修复补丁
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# 1. 模型结构保持不变
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
    print("Lab 09 任务 5-7：验证、测试与绘制训练曲线")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\n🚀 开始包含验证集的完整训练流程...")
    for epoch in range(epochs):
        # --- 任务 4：训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # --- 任务 5：验证阶段 [cite: 129-133] ---
        model.eval() # 切换到评估模式
        val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad(): # 不计算梯度，节省显存加速
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # --- 任务 6：测试阶段 [cite: 135-138] ---
    print("\n🎯 开始在测试集上评估最终模型...")
    model.eval()
    test_loss = 0.0
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    final_test_loss = test_loss / len(test_loader)
    final_test_acc = correct_test / total_test
    print(f"⭐ 测试集最终结果 | Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.4f}")

    # --- 任务 7：绘制曲线图 [cite: 141-146] ---
    print("\n📈 正在生成并保存训练曲线和预测结果图...")
    
    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png', dpi=300)
    plt.close()

    # 绘制 Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), train_accs, label='Training Accuracy', marker='o')
    plt.plot(range(1, epochs+1), val_accs, label='Validation Accuracy', marker='x')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png', dpi=300)
    plt.close()

    # 绘制 8 张测试图像预测结果 [cite: 139-140]
    test_loader_sample = DataLoader(test_dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(test_loader_sample))
    images_gpu = images.to(device)
    outputs = model(images_gpu)
    _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.title(f"True: {labels[i].item()} | Pred: {predicted[i].item()}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=300)
    plt.close()
    
    print("✅ 所有图表已成功保存：'loss_curve.png', 'accuracy_curve.png', 'test_predictions.png'")

if __name__ == "__main__":
    main()