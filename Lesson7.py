"""
畳み込みニューラルネットワークを実装する
"""

import sys, os
sys.path.append(os.pardir)
from common.util import im2col, col2im
import numpy as np

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