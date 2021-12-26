import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

data = np.load('mnist.npz')
# print(data.files)
# 读取数据、处理、挑选0，9、主成分分析
x_train , y_train , x_test , y_test = data['X_train'] , data['y_train'] , data['X_test'] , data['y_test']
x_train.resize((60000,28*28))
x_test.resize((10000,28*28))
y_label = []
x_train_slt = []
for i in range(len(y_train)):
    tmp = np.argmax(y_train[i])
    if tmp == 0:
        y_label.append(0)
        x_train_slt.append(x_train[i])
    elif tmp == 9:
        y_label.append(1)
        x_train_slt.append(x_train[i])
x_train_slt = np.array(x_train_slt)
y_label = np.array(y_label)
y_label = y_label.reshape(-1,1)
N = 10
# PCA
pca = PCA(n_components=N)
newX = pca.fit_transform(x_train_slt)
arr = np.concatenate((newX,y_label),axis=1)
print(arr.shape)
np.random.shuffle(arr)
# arr = arr[:1000,:]
# TSNE
# tsne = TSNE(n_components=N)
# newX = tsne.fit_transform(x_train_slt)
# arr = np.concatenate((newX,y_label),axis=1)
# print(arr.shape)
# np.random.shuffle(arr)
# arr = arr[:1000,:]
# 画图
if N == 2:
    colors = ['b','g']
    for i in range(len(arr)):
        plt.scatter(arr[i,0],arr[i,1],color=colors[int(arr[i,2])],s=5)
plt.show()
# 测试
y_test_label = []
x_test_slt = []
for i in range(len(x_test)):
    tmp = np.argmax(y_test[i])
    if tmp == 0:
        y_test_label.append(0)
        x_test_slt.append(x_test[i])
    elif tmp == 9:
        y_test_label.append(1)
        x_test_slt.append(x_test[i])
x_test_slt = np.array(x_test_slt)
y_test_label = np.array(y_test_label)
# y_test_label = y_test_label.reshape(-1,1)
newX_test = pca.fit_transform(x_test_slt)
mdl = LogisticRegression(max_iter=10000)
mdl.fit(arr[:,:-1],arr[:,-1])
pred = mdl.predict(newX_test)
acc = sum(pred == y_test_label) / len(y_test_label)
print('acc:',acc)
# 不做特征提取时正确率
mdl.fit(x_train_slt,y_label)
pred = mdl.predict(x_test_slt)
acc = sum(pred == y_test_label) / len(y_test_label)
print('all acc:',acc)