"""
畳み込みニューラルネットワークを実装する
"""

import sys, os
sys.path.append(os.pardir)
from common.util import im2col, col2im
import numpy as np
from collections import OrderedDict
from Lesson5 import Relu, Affine, SoftmaxWithLoss

"""
Convolution（畳み込み）レイヤ
"""
class Convolution:
    def __init__(self, W, b, stride=1,pad=0):
        self.W = W # 重み
        self.b = b # バイアス
        self.stride = stride # ストライド
        self.pad = pad # パディング

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride) # 出力データの高さ
        out_w = int(1 + (W * 2 * self.pad - FW) / self.stride) # 出力データの幅

        col = im2col(x, FH, FW, self.stride, self.pad) # 2次元配列に変換
        col_W = self.W.reshape(FN, -1).T # フィルターの展開 # -1で辻褄が合うように自動で計算してくれる
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 4次元配列に変換 #.transpose()は次元の順番を入れ替える

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) # 4次元配列を2次元配列に変換
        
        self.db = np.sum(dout, axis=0) # バイアスの勾配
        self.dW = np.dot(self.col.T, dout) # 重みの勾配 # col.Tはcolの転置
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW) # 重みの勾配

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    

"""
Poolingレイヤ
"""
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h # プーリングの高さ
        self.pool_w = pool_w # プーリングの幅
        self.stride = stride # ストライド
        self.pad = pad # パディング

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride) # 出力データの高さ
        out_w = int(1 + (W - self.pool_w) / self.stride) # 出力データの幅

        # 展開
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 最大値
        out = np.max(col, axis=1) # 最大値を求める
        # 整形
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # 4次元配列に変換

        self.x = x
        self.arg_max = np.argmax(col, axis=1)

        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1) # 4次元配列を2次元配列に変換

        pool_size = self.pool_h * self.pool_w 
        dmax = np.zeros((dout.size, pool_size)) # 0で初期化
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() # flatten()は多次元配列を1次元配列に変換
        dmax = dmax.reshape(dout.shape + (pool_size,)) # 3次元配列に変換

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1) # 2次元配列に変換
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad) # 2次元配列を4次元配列に変換

        return dx
    
"""
CNNの実装
"""
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num'] # フィルターの数
        filter_size = conv_param['filter_size'] # フィルターのサイズ
        filter_pad = conv_param['pad'] # パディング
        filter_stride = conv_param['stride'] # ストライド
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1 # 畳み込み層の出力サイズ
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2)) # プーリング層の出力サイズ

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads
    
    