import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import urllib.request # 新增：用于处理网络请求

# ==========================================
# 🌟 网络修复补丁：伪装成正常的浏览器，防止被服务器拒绝连接
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
urllib.request.install_opener(opener)
# ==========================================

def main():
    print("="*50)
    print("Lab 09 任务 2：加载图像数据集 (MNIST)")
    print("="*50)

    # 1. 定义数据预处理：将图像转换为 PyTorch 可以处理的 Tensor 格式
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 2. 下载并加载数据集
    print("正在下载并加载 MNIST 数据集... (首次运行可能需要一点时间，请耐心等待)")
    # 下载完整的训练集 (共 60000 张)
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 下载测试集 (共 10000 张)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 3. 将完整的训练集进一步划分为：训练集 和 验证集
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"\n✅ 数据加载与划分完成！")
    print(f"   - 训练集样本数: {len(train_dataset)}")
    print(f"   - 验证集样本数: {len(val_dataset)}")
    print(f"   - 测试集样本数: {len(test_dataset)}")

    # 4. 显示至少 8 张样本图像并标注真实类别
    print("\n正在生成 8 张样本图像并保存...")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True Label: {labels[i].item()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=300)
    print("✅ 样本图像已保存为 'mnist_samples.png'！")
    
    # 因为是在 WSL 环境下，为了防止弹窗卡死，我们注释掉 plt.show()，直接看保存的图片即可
    # plt.show() 

if __name__ == "__main__":
    main()