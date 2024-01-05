# newral network
import numpy as np
import matplotlib.pylab as plt

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
B2 = np.array([0.1, 0.2]) # ３層目のバイアス（1 * 2行列）

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2 # 重み付き信号とバイアスの総和
Z2 = sigmoid(A2) # 第２層（1 * 2行列）

print(A2)
print(Z2)

#恒等関数
def identity_function(x):
    return x

W3 = np.array([[0.3, 0.5],[0.1, 0.4]])
B3 = np.array([0.1, 0.3])

print(Z2.shape)
print(W3.shape)
print(B3.shape)

A3 = np.dot(Z2, W3) + B3
Z3 = identity_function(A3)

print(A3)
print(Z3)

