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
