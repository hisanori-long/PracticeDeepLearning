# Description: This file contains the code for Lesson 1 of ゼロから作るDeepLearning 

class UserClass:
    def __init__(self, name, age): # Constructor
        self.name = name
        self.age = age
        print("User created!")

    def say_hello(self):
        print(f"Hello, {self.name}!")

user1 = UserClass("John", 36)
user1.say_hello()

