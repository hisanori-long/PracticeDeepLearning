import numpy as np
import matplotlib.pyplot as plt

"""
Momentum
"""
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum # 空気抵抗
        self.v = None # 速度

    def update(self, params, grads):
        if self.v is None: # 初期設定
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.le * grads[key]
            params[key] += self.v[key]

"""
AdaGrad
"""
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None: # 初期設定
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # hに勾配の二乗を加算し、更新
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) # 1e-7は0除算を防ぐための微小値


"""
アクティベーションの分布を確認
"""
# 重みの初期値の分散によって、アクティベーションの分布がどのように変化するかを確認
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.random.randn(1000, 100) # 1000個のデータ
node_num =100 # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5 # 隠れ層が5層
activations = {} # ここにアクティベーションの結果を格納

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # w = np.random.randn(node_num, node_num) * 1 # 標準偏差が1
    # w = np.random.randn(node_num, node_num) * 0.01 # 標準偏差が0.01
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num) # Xavierの初期値
    
    z = np.dot(x, w)
    a = sigmoid(z)
    # a = tanh(z)
    activations[i] = a

# ヒストグラムを描画
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()