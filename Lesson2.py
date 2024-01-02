# learning perceptron

# AND gate perceptron 両方の入力が1の時だけ１を出力する
def AND(x1, x2):
    w1, w2, theta = 0.3, 0.3, 0.4
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
print("AND gate perceptron")
print(AND(1, 1))
print(AND(1, 0))
print(AND(0, 1))
print(AND(0, 0))


def NAND(x1, x2):
    w1, w2, theta = -0.3, -0.3, -0.4
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print("\n NAND gate perceptron") 
print(NAND(1, 1))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(0, 0))

def OR(x1, x2):
    w1, w2, theta = 0.3, 0.3, 0.2
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print("\n OR gate perceptron")
print(OR(1,1))
print(OR(1, 0))
print(OR(0, 1))
print(OR(0, 0))

# using bias
import numpy as np
def AND_bias(x1, x2):
    x = np.array([x1, x2]) # input 
    w = np.array([0.3, 0.3]) # weight 入力信号に対する重要度のコントロール
    b = -0.4 # bias ニューロンの発火のしやすさ
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    
print("\n AND gate perceptron using bias")
print(AND_bias(1, 1))
print(AND_bias(1, 0))
print(AND_bias(0, 1))
print(AND_bias(0, 0))

def NAND_bias(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.3, -0.3])
    b = 0.4
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    
print("\n NAND gate perceptron using bias")
print(NAND_bias(1, 1))
print(NAND_bias(1, 0))
print(NAND_bias(0, 1))
print(NAND_bias(0, 0))

def OR_bias(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.3, -0.3])
    b = 0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1
    
print("\n OR gate perceptron using bias")
print(OR_bias(1, 1))
print(OR_bias(1, 0))
print(OR_bias(0, 1))
print(OR_bias(0, 0))


    