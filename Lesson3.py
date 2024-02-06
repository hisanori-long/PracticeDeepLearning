# newral network
import numpy as np
import matplotlib.pylab as plt
import sys, os
from PIL import Image

# ステップ関数
# パーセプトロンでは、ステップ関数を活性化関数として使用
def step_function(x):
    #もっと簡潔に書くなら： return np.array(x > 0, dtype=np.int)
    y = x > 0 # xが0以上ならTrue, 0以下ならFalseを返す処理
    return y.astype(np.int) # bool型をint型に変換
    
x = np.arange(-5.0, 5.0, 0.1) # -5.0から5.0まで0.1刻みの配列を生成
y1 = step_function(np.array(x))
plt.plot(x, y1, linestyle= "--")
# plt.ylim(-0.1, 1.1) # y軸の範囲を指定
# plt.show()

# シグモイド関数
# ニューラルネットワークでは、シグモイド関数を活性化関数として使用
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y2 = sigmoid(x)
plt.plot(x, y2)
# plt.ylim(-0.1, 1.1)
# plt.show()

# ReLU関数
# 最近は、ReLU関数がニューラルネットワークで用いられている
def relu(x):
    return np.maximum(0, x) # maximumでは、値の大きい方を出力する（x < 0の時は、0が大きい、 0 <= 0の時は、xが大きい）

y3 = relu(x)
plt.plot(x, y3)
plt.ylim(-0.1, 1.1)
# plt.show()


#行列を用いた３層のニューラルネットワークの作成
print(" \n 3層のニューラルネットワークの実装")
X = np.array([1.0, 0.5]) # 入力層（1 * 2行列）
W1 = np.array([[0.1, 0.3, 0.4], [0.5, 1.0, 0.2]]) # 1層目の重み （2 * 3 行列）
B1 = np.array([0.1, 0.2, 0.3]) # 1層目のバイアス(1 * 3行列)

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1 # 重み付き信号とバイアスの総和
Z1 = sigmoid(A1) # 第１層（1 * 3行列）

print(A1)
print(Z1)

W2 = np.array([[0.3, 0.7],[1.0, 0.7],[0.8, 0.2]]) #２層目の重み（3 * 2行列）
B2 = np.array([0.1, 0.2]) # ２層目のバイアス（1 * 2行列）

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2 # 重み付き信号とバイアスの総和
Z2 = sigmoid(A2) # 第２層（1 * 2行列）

print(A2)
print(Z2)

#恒等関数（今回はそのまま返す）
def identity_function(x):
    return x

W3 = np.array([[0.3, 0.5],[0.1, 0.4]]) # ３層目の重み（2 * 2行列）
B3 = np.array([0.1, 0.3]) # 3層目のバイアス（1 * 2行列）

print(Z2.shape)
print(W3.shape)
print(B3.shape)

A3 = np.dot(Z2, W3) + B3 # 重み付き信号とバイアスの総和
Z3 = identity_function(A3) #出力層（1 * 2行列）

print(A3)
print(Z3)

#先ほどのニューラルネットワークを見やすくする
print("\n 簡潔にしたニューラルネットワーク")
# 各層の重みとバイアスを定義
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.4], [0.5, 1.0, 0.2]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.3, 0.7],[1.0, 0.7],[0.8, 0.2]])
    network['B2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.3, 0.5],[0.1, 0.4]])
    network['B3'] = np.array([0.1, 0.3])

    return network

# 各層を求める（入力から出力にかけて計算するため、forward）
def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5]) 
y = forward(network, x)
print(y)


# ソフトマックス関数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 各入力を指数関数でとる（a - c で値が大きくなり、オーバーフローを起こすのを防ぐ）
    sum_exp_a = np.sum(exp_a) # 指数関数でとった入力の総和
    y = exp_a / sum_exp_a # 各入力を総和で割る
    return y


#手書き数字認識
import sys, os
sys.path.append('../deep-learning-from-scratch-master') # 親ディレクトリのファイルをインポートするための設定
from sklearn.datasets import load_digits # sklearnを使用して、mnistのデータセットを使用
from sklearn.model_selection import train_test_split # sklearnを使用して、データセットを訓練データとテストデータに分割
# from dataset.mnist import load_mnist

#データの取得、格納
digits = load_digits()
data = digits.data
target = digits.target

# データの分割
x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=0.2, random_state=0)

# データの正規化
x_train = x_train / 255.0
x_test = x_test / 255.0
t_train = t_train / 255.0
t_test = t_test / 255.0

#データの表示
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
img = img.reshape(8, 8)
print(img.shape)
img_show(img)