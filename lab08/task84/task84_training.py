from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 导入五种以上的模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. 加载并划分数据
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=42
    )

    # 2. 定义模型字典
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "SVM": SVC(kernel='rbf'),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    print(f"{'模型名称':<20} | {'测试准确率 (Accuracy)':<15}")
    print("-" * 45)

    # 3. 循环训练并评估
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)          # 训练模型
        y_pred = model.predict(X_test)       # 预测
        acc = accuracy_score(y_test, y_pred) # 计算准确率
        results[name] = acc
        print(f"{name:<20} | {acc:.4f}")

    print("-" * 45)
    print("✅ 所有模型训练与评估完成！")

if __name__ == "__main__":
    main()