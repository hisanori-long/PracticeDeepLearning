import numpy as np

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
