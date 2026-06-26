import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    print("="*50)
    print("Lab 08 任务 6：错误样本分析")
    print("="*50)

    # 1. 加载数据（同时加载 data 和 images 以便后续画图）
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        digits.data, digits.target, digits.images, test_size=0.25, random_state=42
    )

    # 2. 训练表现最好的 SVM 模型
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 3. 绘制并保存混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("SVM Confusion Matrix")
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ 混淆矩阵已保存为 'confusion_matrix.png'")
    plt.close()

    # 4. 找出错误分类的样本索引
    error_indices = np.where(y_pred != y_test)[0]
    print(f"测试集中共有 {len(error_indices)} 个错误预测样本。")

    # 5. 绘制并保存前 5 个错误样本
    plt.figure(figsize=(10, 3))
    for i, err_idx in enumerate(error_indices[:5]): # 取前5个
        plt.subplot(1, 5, i + 1)
        plt.imshow(img_test[err_idx], cmap=plt.cm.gray_r, interpolation='nearest')
        
        # 标出真实标签和预测标签
        true_label = y_test[err_idx]
        pred_label = y_pred[err_idx]
        plt.title(f"True: {true_label}\nPred: {pred_label}", color="red")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('error_samples.png', dpi=300)
    print("✅ 错误样本图已保存为 'error_samples.png'")
    
    print("\n任务 6 运行完毕！")

if __name__ == "__main__":
    main()