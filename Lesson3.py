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
plt.show()

