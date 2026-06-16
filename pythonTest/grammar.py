from typing import List

if __name__ == '__main__':
    scope = 60
    if scope > 60:
        print("pass");
    elif scope < 60:
        print("noPass");
    else:
        print("NONE");

# java -> new List<String>;
pythonArr = ["张三", "李四"]
# java -> Map<"key","value">
pythonMap = {"k": "v", "k1": "v1"}
# java -> Set<String>
pythoneUnique = {"set1", "set2", "set3"}
# jav  -> 无 python 元数组,只读
pythonTuple = ("1", "2")


def add(a, b):
    return a + b;


def get_user_info():
    return "Null", 25


name, age = get_user_info();

print(name, age);


class Person:
    def __init__(self,name,age):
        # 成员变量不需要提前声明，直接赋值即可
        self.name = name
        # 双下划线开头表示私有变量 (private)
        self.__age = age
        # 成员方法
    def say_Hi(self):
        print(f"HI ,i am {self.name}")


p = Person("Tom",18)
p.say_Hi()


def test_F_Loss():
    print("F String 的测试,还有F .: 测试损失度")
    name = "NUll"
    loss = 3.1415926
    print(f"name: {name+'this is nb'}, loss: {loss:.4F}")


test_F_Loss()