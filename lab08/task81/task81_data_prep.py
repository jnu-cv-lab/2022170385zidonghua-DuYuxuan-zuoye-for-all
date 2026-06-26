from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("="*50)
    print("Lab 08 任务 1：数据准备")
    print("="*50)

    # 1. 加载手写数字图像数据集
    digits = datasets.load_digits()

    # 2. 查看数据集中图像的数量
    # digits.images 是一个三维数组 (样本数, 高, 宽)
    num_images = len(digits.images)
    print(f"1. 数据集中图像的数量为: {num_images} 张")

    # 3. 查看每张图像的大小
    # 取第一张图像查看其 shape
    image_shape = digits.images[0].shape
    print(f"2. 每张图像的大小为: {image_shape[0]} × {image_shape[1]} 像素")

    # 4. 查看类别标签
    # digits.target 包含了所有图像的对应标签
    unique_labels = np.unique(digits.target)
    print(f"3. 类别标签包含: {unique_labels}，共 {len(unique_labels)} 类")

    # 5. 显示若干张样本图像及其真实标签
    print("4. 正在生成样本图像并保存...")
    # 创建一个图形窗口，显示前 10 张图片
    plt.figure(figsize=(12, 5))
    for index, (image, label) in enumerate(zip(digits.images[:10], digits.target[:10])):
        plt.subplot(2, 5, index + 1)
        # 使用灰度反转的颜色映射 (白底黑字) 更符合手写习惯
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f"True Label: {label}", fontsize=12)
        plt.axis('off') # 不显示坐标轴

    plt.tight_layout()
    # 将生成的图片保存下来，方便你之后直接插入到 Word/PDF 实验报告中
    plt.savefig('sample_digits.png', dpi=300)
    print("✅ 样本图像已保存为 'sample_digits.png'！")
    
    # 弹出窗口显示图像
    plt.show()

if __name__ == "__main__":
    main()