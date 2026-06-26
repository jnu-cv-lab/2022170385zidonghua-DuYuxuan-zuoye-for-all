from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    print("="*50)
    print("Lab 08 任务 2：数据划分")
    print("="*50)

    # 1. 加载数据
    digits = datasets.load_digits()
    X = digits.data    # 特征数据 (1797, 64)
    y = digits.target  # 标签数据 (1797,)

    # 2. 划分训练集和测试集 (测试集比例 25%)
    # random_state=42 保证每次划分的结果一样，方便复现实验结果
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 3. 打印划分结果
    print(f"数据集总样本数: {len(X)} 张")
    print(f"训练集样本数: {len(X_train)} 张 (占比 {len(X_train)/len(X)*100:.1f}%)")
    print(f"测试集样本数: {len(X_test)} 张 (占比 {len(X_test)/len(X)*100:.1f}%)")
    
    print("\n✅ 数据划分成功！")

if __name__ == "__main__":
    main()