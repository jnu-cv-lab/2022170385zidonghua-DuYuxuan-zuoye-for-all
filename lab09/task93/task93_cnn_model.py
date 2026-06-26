import torch
import torch.nn as nn

# 定义一个简单的卷积神经网络 (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一层卷积：输入 1 个通道 (灰度图)，输出 16 个特征图，卷积核大小 3x3，填充 1 保持尺寸
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU() # 激活函数
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 池化层，尺寸减半变为 14x14

        # 第二层卷积：输入 16 个通道，输出 32 个特征图，卷积核大小 3x3，填充 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 池化层，尺寸再次减半变为 7x7

        # 全连接层 (Flatten 之后)
        # 输入特征数为：32 (通道数) * 7 (宽) * 7 (高) = 1568
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        
        # 输出层：输出 10 个类别 (数字 0-9)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 定义数据在网络中的前向传播过程
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # 展平特征图以送入全连接层
        x = x.view(-1, 32 * 7 * 7) 
        
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print("="*50)
    print("Lab 09 任务 3：定义 CNN 模型结构")
    print("="*50)

    # 实例化模型
    model = SimpleCNN()
    
    # 打印模型结构
    print(model)
    print("\n✅ 模型定义成功！该结构符合 MNIST 28x28 灰度图的输入，并输出 10 个类别。")

if __name__ == "__main__":
    main()