from sklearn import datasets

def main():
    print("="*50)
    print("Lab 08 任务 3：特征表示验证")
    print("="*50)

    digits = datasets.load_digits()
    
    # 获取第一张图片的原始 2D 矩阵形式
    image_2d = digits.images[0]
    print(f"1. 原始图像 (digits.images[0]) 的维度是: {image_2d.shape}")
    print("   它是一个 8 行 8 列的二维矩阵。")
    
    # 获取第一张图片作为特征向量的 1D 形式
    vector_1d = digits.data[0]
    print(f"\n2. 转换后的特征向量 (digits.data[0]) 的维度是: {vector_1d.shape}")
    print("   它已经被展平成了一个包含 64 个元素的一维数组。")
    
    print("\n✅ 特征表示验证完成！")

if __name__ == "__main__":
    main()