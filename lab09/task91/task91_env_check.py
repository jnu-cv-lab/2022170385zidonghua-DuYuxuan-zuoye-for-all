import torch
import torchvision
import numpy as np
import matplotlib

def main():
    print("="*50)
    print("Lab 09 任务 1：PyTorch 环境测试")
    print("="*50)
    
    # 测试库的导入与版本
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Torchvision 版本: {torchvision.__version__}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"Matplotlib 版本: {matplotlib.__version__}")

    # 判断当前环境是否支持 GPU (CUDA)
    print("\n" + "-"*20)
    cuda_available = torch.cuda.is_available()
    print(f"GPU (CUDA) 是否可用: {cuda_available}")
    if cuda_available:
        print(f"当前使用的 GPU 设备: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到可用 GPU，后续计算将使用 CPU 进行。")

    # 简单的 PyTorch 张量操作
    print("\n" + "-"*20)
    print("PyTorch 张量 (Tensor) 操作测试：")
    tensor_a = torch.tensor([[1, 2], [3, 4]])
    tensor_b = torch.tensor([[5, 6], [7, 8]])
    tensor_c = tensor_a + tensor_b
    
    print(f"Tensor A:\n{tensor_a}")
    print(f"Tensor B:\n{tensor_b}")
    print(f"A + B 的结果:\n{tensor_c}")
    
    print("\n✅ 任务 1：环境测试全部通过！")

if __name__ == "__main__":
    main()