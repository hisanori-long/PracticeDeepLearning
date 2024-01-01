# Description: This file contains the code for Lesson 1 of ゼロから作るDeepLearning 

class UserClass:
    def __init__(self, name, age): # Constructor クラス作成次に実行される
        self.name = name
        self.age = age
        print("User created!")

    def say_hello(self): # Class method クラス内で定義される関数
        print(f"Hello, {self.name}!")

user1 = UserClass("John", 36) # クラスのインスタンス化 init関数のnameとageの引数を渡す
user1.say_hello()

# practice numpy
print("\n Practice numpy" )
import numpy as np # numpyのインポート

x = np.array([1.0, 5.0, 10.0]) #numpy配列を作成
y = np.array([10.0, 8.0, -2.0])

# 四則演算
print(x + y)
print(x - y)
print(x * y)
print(x / y)

print("\n 2*2 matrix")
A = np.array([[10.0, 15.0], [2.0, 7.0]]) # 2*2 配列
B = np.array([[9.0, 10.0], [3.0, 5.0]])

print(A.shape)
print(A.dtype)


print(A + B)
print(A - B)
print(A * B)
print(A / B)

# broadcast
print("\n broadcast")
A = np.array([[10.0, 15.0], [2.0, 7.0]])
B = np.array([8.0, 10.0])
print(A * 10) 
print(A * B)

print(A[0])
print(A[0][1])
print()

for row in A:
    print(row)

print()
A = A.flatten() # 多次元配列を1次元配列に変換
print(A)

print(A[np.array([0, 2, 3])]) # １次元配列の特定の要素を取得

print()
print(A > 6) # 条件が合う要素をTrueで返す
print(A[A > 6]) # 条件があう要素を取得




