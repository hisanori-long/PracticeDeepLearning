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

def cross_entropy_error_batch(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
