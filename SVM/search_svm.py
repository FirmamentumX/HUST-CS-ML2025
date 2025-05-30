from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import joblib
import numpy as np

# 加载数据
transform = transforms.ToTensor()
train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 转换为numpy数组并展平，进行像素归一化
X_train = train_set.data.numpy().reshape(60000, -1) / 255.0
y_train = train_set.targets.numpy()
X_test = test_set.data.numpy().reshape(10000, -1) / 255.0
y_test = test_set.targets.numpy()

# 进行标准(正态分布)化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 网格搜索最优化超参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 使用最优参数训练最终模型
best_params = grid_search.best_params_
final_model = svm.SVC(**best_params)
final_model.fit(X_train_scaled, y_train)

# 测试集评估
y_pred = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

# 打印结果
print("Best parameters:", best_params)
print("Validation accuracy:", grid_search.best_score_)
print("Test accuracy:", test_accuracy)

# 保存模型
joblib.dump(final_model, 'svm_model_best.pkl')