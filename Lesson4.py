import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

"""
損失関数を定義
"""
# 二乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2) # a**bは aのb乗と同義

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7 # 1 × 10^-7と同義
    return -np.sum(t * np.log(y + delta)) 

"""
ミニバッチ学習
"""
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # train_size(60000)からbatch_size(10)をランダムに選ぶ
x_batch = x_train[batch_mask]
t_batch = x_train[batch_mask]

print('batch_mask', batch_mask)

#バッチ対応した、交差エントロピー誤差
def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

"""
学習アルゴリズムの実装
"""

import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #重みの初期化
        self.params = {}
        # 0 ~　1 の標準化されたランダムな数字を(input_size × hidden_size)の2次元配列に格納する
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 0をhidden_sizeの１次元配列に格納する
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x:入力データ t:教師データ
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

        return grads
    
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print('shape W1' ,net.params['W1'].shape)
print('shape b1', net.params['b1'].shape)
print('shape W2', net.params['W2'].shape)
print('shape b2', net.params['b2'].shape)

"""
ミニバッチ学習の実装
"""
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

# ハイバーパラメータの設定
iters_num = 10000 # 学習回数
train_size = x_train.shape[0]
batch_size = 100 # バッチ処理のデータ数
learning_rate = 0.1 # パラメータの変化度合い

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    # grad = network.numerical_gradient(x_batch, t_batch) 
    # 処理に時間がかかるので、高速に勾配を求める
    grad = network.gradient(x_batch, t_batch)

    # パラメータを修正
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックごとの精度をKさん
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + "," + str(test_acc))


