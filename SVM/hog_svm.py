from skimage.feature import hog
from sklearn.utils import resample
from sklearn import svm
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
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

# 将图像恢复为28x28格式
X_train_2d = X_train_scaled.reshape(-1, 28, 28)
X_test_2d = X_test_scaled.reshape(-1, 28, 28)

# 提取 HOG 特征
X_train_hog = [hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2)) for img in X_train_2d]
X_test_hog = [hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2)) for img in X_test_2d]

# 训练 SVM
clf = svm.SVC(kernel='rbf', C=10, gamma=0.01, verbose=True).fit(X_train_hog, y_train)


# 评估
pred = clf.predict(X_test_hog)
print(f"SVM Accuracy: {accuracy_score(y_test, pred):.4f}")

# 保存模型
joblib.dump(clf, './models/hog_svm.pkl')
# [LibSVM]SVM Accuracy: 0.8940