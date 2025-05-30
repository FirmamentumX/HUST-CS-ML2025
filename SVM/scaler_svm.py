from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from torchvision import datasets, transforms
import joblib

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

# 训练
clf = svm.SVC(kernel='rbf', C=10, gamma=0.01, verbose=True)
clf.fit(X_train_scaled, y_train)

# 评估
pred = clf.predict(X_test_scaled)
print(f"SVM Accuracy: {accuracy_score(y_test, pred):.4f}, F1 Score: {f1_score(y_test, pred, average='macro'):.4f}")

# 保存模型
joblib.dump(clf, './models/scaler_svm.pkl')
# [LibSVM]SVM Accuracy: 0.8999, F1 Score: 0.8998