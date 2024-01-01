# learning perceptron

# AND gate perceptron 両方の入力が1の時だけ１を出力する
def AND(x1, x2):
    w1, w2, theta = 0.3, 0.3, 0.4
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(AND(1, 1))
print(AND(1, 0))
print(AND(0, 1))
print(AND(0, 0))