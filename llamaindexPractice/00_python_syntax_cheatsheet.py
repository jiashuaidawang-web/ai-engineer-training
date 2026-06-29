"""
00_python_syntax_cheatsheet.py — Python 语法速查（Java 程序员视角）

这个文件是 Python 语法的速查手册，帮你快速跨越语法障碍。
每个知识点都附带 Java 类比，让你用已有的 Java 知识理解 Python。

运行方式: python 00_python_syntax_cheatsheet.py
"""

# ============================================================
# 【Python 语法】注释
#   # 单行注释
#   """ 多行注释（也叫 docstring，用于文档字符串）
#   Java: // 单行  /* 多行 */
# ============================================================

# ============================================================
# 1. 变量和数据类型
# ============================================================

# --- 【Python 语法】动态类型 ---
# Python 不需要声明变量类型，赋值即创建
# 类比 Java: int x = 10;  →  Python: x = 10
x = 10              # int（整数）
y = 3.14            # float（浮点数）
name = "Hello"      # str（字符串）
is_valid = True     # bool（布尔值，注意首字母大写 True/False，Java 是 true/false）
nothing = None      # None 相当于 Java 的 null

# --- 【Python 语法】f-string（格式化字符串）---
# f"..." 中的 {变量} 会被替换为变量的值
# 类比 Java: "Hello, " + name + "!"
# 或者 Java 15+: "Hello, %s!".formatted(name)
print(f"x = {x}, y = {y}, name = {name}")

# --- 【Python 语法】字符串操作 ---
# Python 的字符串操作比 Java 简洁得多
text = "  Hello World  "
print(f"去掉空格: '{text.strip()}'")       # 去掉两端空格（类比 Java: text.trim()）
print(f"大写: '{text.upper()}'")           # 全部大写（类比 Java: text.toUpperCase()）
print(f"替换: '{text.replace('World', 'Python')}'")  # 替换（类比 Java: text.replace(...)）
print(f"分割: {text.split()}")             # 分割成列表（类比 Java: text.split("\\s+")）
print(f"切片 [0:5]: '{text.strip()[0:5]}'")  # 截取子串（类比 Java: text.substring(0, 5)）

# --- 【Python 语法】列表（List）---
# Python 的 list 类比 Java 的 ArrayList
# Java: List<String> fruits = Arrays.asList("apple", "banana", "cherry");
fruits = ["apple", "banana", "cherry"]
print(f"\n列表: {fruits}")
print(f"长度: {len(fruits)}")              # 类比 Java: fruits.size()
print(f"第一个: {fruits[0]}")              # 索引从 0 开始（类比 Java: fruits.get(0)）
print(f"最后一个: {fruits[-1]}")           # -1 表示最后一个（Java 没有这个语法）
print(f"切片 [1:3]: {fruits[1:3]}")        # 左闭右开 [1, 3)（类比 Java: fruits.subList(1, 3)）
fruits.append("date")                      # 添加元素（类比 Java: fruits.add("date")）
print(f"添加后: {fruits}")

# --- 【Python 语法】字典（Dictionary）---
# Python 的 dict 类比 Java 的 HashMap
# Java: Map<String, Integer> ages = new HashMap<>(); ages.put("Alice", 25);
ages = {"Alice": 25, "Bob": 30, "Charlie": 35}
print(f"\n字典: {ages}")
print(f"Alice 的年龄: {ages['Alice']}")    # 取值（类比 Java: ages.get("Alice")）
print(f"所有名字: {list(ages.keys())}")    # 所有键（类比 Java: ages.keySet()）
print(f"所有年龄: {list(ages.values())}")  # 所有值（类比 Java: ages.values()）
ages["Dave"] = 40                          # 添加/修改（类比 Java: ages.put("Dave", 40)）
print(f"添加后: {ages}")
print(f"'Eve' 在字典中吗？{'Eve' in ages}")  # 检查键是否存在（类比 Java: ages.containsKey("Eve")）

# --- 【Python 语法】元组（Tuple）---
# 不可变的列表，一旦创建不能修改
# 类比 Java: 没有直接等价物，最接近的是 final List<T> 或 record
point = (10, 20)
# point[0] = 30  # ❌ 这会报错！元组不可变
print(f"\n元组: {point}")

# --- 【Python 语法】集合（Set）---
# 不重复的元素集合
# 类比 Java: HashSet
unique_numbers = {1, 2, 3, 3, 4, 5}  # 自动去重
print(f"集合: {unique_numbers}")  # 输出 {1, 2, 3, 4, 5}

# ============================================================
# 2. 控制流
# ============================================================

# --- 【Python 语法】if/elif/else ---
# 注意：用冒号 : 和缩进代替了大括号 {}
age = 25
if age < 18:
    print("未成年人")
elif age < 65:          # elif 类比 Java 的 else if
    print("成年人")
else:
    print("老年人")

# --- 【Python 语法】三元表达式 ---
# 条件 if 条件成立 else 条件不成立
# 类比 Java: age >= 18 ? "成年" : "未成年"
status = "成年" if age >= 18 else "未成年"
print(f"状态: {status}")

# --- 【Python 语法】for 循环 ---
# Python 的 for 是「增强型 for 循环」（foreach）
# 类比 Java: for (String fruit : fruits) { ... }
print("\n水果列表:")
for fruit in fruits:
    print(f"  - {fruit}")

# --- 【Python 语法】range() ---
# range(n) 生成 0, 1, 2, ..., n-1
# range(start, end) 生成 start, start+1, ..., end-1
# range(start, end, step) 生成 start, start+step, start+2*step, ...
# 类比 Java: for (int i = 0; i < 5; i++)
print("\nrange(5):")
for i in range(5):
    print(f"  {i}", end=" ")  # end=" " 表示不换行，用空格结尾
print()  # 换行

print("range(2, 6):")
for i in range(2, 6):
    print(f"  {i}", end=" ")
print()

print("range(0, 10, 2):")
for i in range(0, 10, 2):
    print(f"  {i}", end=" ")
print()

# --- 【Python 语法】while 循环 ---
count = 0
while count < 3:
    print(f"\nwhile 循环: count = {count}")
    count += 1  # Python 没有 ++ 运算符，用 += 1 代替

# ============================================================
# 3. 函数
# ============================================================

# --- 【Python 语法】def 定义函数 ---
# 不需要写返回值类型和访问修饰符
# 类比 Java: public int add(int a, int b) { return a + b; }
def add(a, b):
    """
    这是一个函数的文档字符串（docstring）
    类比 Java 的 Javadoc: /** ... */
    """
    return a + b

print(f"\nadd(3, 5) = {add(3, 5)}")

# --- 【Python 语法】默认参数 ---
# 类比 Java 的方法重载: void greet(String name) 和 void greet(String name, String greeting)
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))                # Hello, Alice!
print(greet("Alice", "Hi"))          # Hi, Alice!

# --- 【Python 语法】可变参数 ---
# *args: 接收任意数量的位置参数，打包成 tuple
# **kwargs: 接收任意数量的关键字参数，打包成 dict
# 类比 Java: public void method(Object... args)
def sum_all(*args):
    """求和任意数量的数字"""
    return sum(args)  # sum() 是 Python 内置函数

print(f"\nsum_all(1, 2, 3, 4) = {sum_all(1, 2, 3, 4)}")

def print_info(**kwargs):
    """打印任意数量的键值对"""
    for key, value in kwargs.items():
        print(f"  {key} = {value}")

print_info(name="Alice", age=25, city="Beijing")

# --- 【Python 语法】lambda 表达式 ---
# 匿名函数，类比 Java 的 Lambda: (a, b) -> a + b
multiply = lambda a, b: a * b
print(f"\nlambda: multiply(3, 4) = {multiply(3, 4)}")

# --- 【Python 语法】列表推导式（List Comprehension）---
# 一行代码完成循环 + 收集，非常强大
# 类比 Java: list.stream().map(x -> x * 2).collect(Collectors.toList())
numbers = [1, 2, 3, 4, 5]
squares = [x ** 2 for x in numbers]  # ** 是幂运算（x 的平方）
print(f"平方: {squares}")  # [1, 4, 9, 16, 25]

# 带条件的列表推导式
# 类比 Java: list.stream().filter(x -> x % 2 == 0).collect(Collectors.toList())
evens = [x for x in numbers if x % 2 == 0]
print(f"偶数: {evens}")  # [2, 4]

# --- 【Python 语法】字典推导式 ---
# 类比 Java: map.entrySet().stream().collect(...)
word_lengths = {word: len(word) for word in ["apple", "banana", "cherry"]}
print(f"\n单词长度: {word_lengths}")  # {'apple': 5, 'banana': 6, 'cherry': 6}

# ============================================================
# 4. 类和面向对象
# ============================================================

# --- 【Python 语法】class 定义类 ---
# __init__ 是构造函数，self 相当于 Java 的 this
# 类比 Java:
#   public class Person {
#       private String name;
#       private int age;
#       public Person(String name, int age) { this.name = name; this.age = age; }
#       public String getInfo() { return name + ", " + age; }
#   }
class Person:
    """
    人类 — 类比 Java 的 class Person
    """
    # --- 【Python 语法】类变量 ---
    # 所有实例共享，类比 Java 的 static 字段
    species = "Homo sapiens"

    def __init__(self, name, age):  # 构造函数（类比 Java: public Person(String name, int age)）
        # --- 【Python 语法】实例变量 ---
        # 类比 Java 的实例字段（private 靠约定，没有真正的访问控制）
        self.name = name  # self.name 类比 Java: this.name
        self.age = age

    def greet(self):  # 实例方法，第一个参数必须是 self（类比 Java: this）
        """打招呼 — 类比 Java: public String greet()"""
        return f"Hi, I'm {self.name}, {self.age} years old."

    def __str__(self):  # 魔术方法，控制 print(obj) 的输出
        """字符串表示 — 类比 Java: @Override public String toString()"""
        return f"Person(name={self.name}, age={self.age})"


# 创建对象（不需要 new 关键字！）
alice = Person("Alice", 25)  # 类比 Java: Person alice = new Person("Alice", 25);
print(f"\n对象: {alice}")          # 调用 __str__
print(f"打招呼: {alice.greet()}")  # 调用实例方法
print(f"物种: {alice.species}")    # 访问实例变量
print(f"类变量: {Person.species}") # 通过类名访问类变量

# --- 【Python 语法】继承 ---
class Student(Person):  # Student 继承 Person（类比 Java: class Student extends Person）
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # 调用父类构造函数（类比 Java: super(name, age)）
        self.student_id = student_id

    def greet(self):  # 重写父类方法（类比 Java: @Override）
        base_greeting = super().greet()  # 调用父类方法
        return f"{base_greeting}, ID: {self.student_id}"

bob = Student("Bob", 20, "S001")
print(f"\n学生: {bob}")
print(f"学生打招呼: {bob.greet()}")

# ============================================================
# 5. 文件操作
# ============================================================

# --- 【Python 语法】with 语句（上下文管理器）---
# 自动关闭文件，不需要 try-finally
# 类比 Java: try (FileReader fr = new FileReader("file.txt")) { ... }
test_file = "test_output.txt"
with open(test_file, "w", encoding="utf-8") as f:  # "w" = 写入模式
    f.write("Hello, Python!\n")
    f.write("This is a test file.\n")

# 读取文件
with open(test_file, "r", encoding="utf-8") as f:  # "r" = 读取模式
    content = f.read()  # 一次性读全部内容（类比 Java: Files.readString(path)）
    print(f"\n文件内容:\n{content}")

# --- 【Python 语法】逐行读取 ---
with open(test_file, "r", encoding="utf-8") as f:
    lines = f.readlines()  # 返回所有行的列表
    print(f"行数: {len(lines)}")

# 清理测试文件
import os
os.remove(test_file)

# ============================================================
# 6. 异常处理
# ============================================================

# --- 【Python 语法】try/except/finally ---
# 类比 Java: try { ... } catch (Exception e) { ... } finally { ... }
try:
    result = 10 / 0
except ZeroDivisionError as e:  # 捕获特定异常（类比 Java: catch (ArithmeticException e)）
    print(f"\n异常: {e}")
except Exception as e:  # 捕获所有异常（类比 Java: catch (Exception e)）
    print(f"其他异常: {e}")
else:  # 没有异常时执行（Java 没有这个语法）
    print("没有异常")
finally:  # 无论是否有异常都会执行（类比 Java: finally）
    print("清理工作")

# ============================================================
# 7. 模块和包
# ============================================================

# --- 【Python 语法】import ---
# import 模块名 — 类比 Java: import java.util.List;
import math
print(f"\nmath.sqrt(16) = {math.sqrt(16)}")

# from 模块名 import 函数名 — 类比 Java: import static java.lang.Math.sqrt;
from math import sqrt, pi
print(f"sqrt(25) = {sqrt(25)}, pi = {pi}")

# import 模块名 as 别名 — 类比 Java: 无（Java 不允许取别名）
import datetime as dt
now = dt.datetime.now()
print(f"当前时间: {now}")

# --- 【Python 语法】__name__ == "__main__" ---
# 这是 Python 的入口点写法
# 类比 Java: public static void main(String[] args)
# __name__ 是内置变量：直接运行此文件时为 "__main__"，被 import 时为文件名
print(f"\n当前模块名: {__name__}")

# ============================================================
# 8. 常用内置函数速查
# ============================================================

# len() — 获取长度（类比 Java: array.length / list.size() / string.length()）
print(f"\nlen([1,2,3]) = {len([1, 2, 3])}")
print(f"len('hello') = {len('hello')}")

# type() — 获取类型（类比 Java: obj.getClass()）
print(f"type(42) = {type(42)}")
print(f"type('hello') = {type('hello')}")
print(f"type([1,2,3]) = {type([1, 2, 3])}")

# isinstance() — 类型检查（类比 Java: obj instanceof Type）
print(f"isinstance(42, int) = {isinstance(42, int)}")
print(f"isinstance('hello', str) = {isinstance('hello', str)}")

# sorted() — 排序（类比 Java: Collections.sort(list)）
print(f"sorted([3,1,4,1,5]) = {sorted([3, 1, 4, 1, 5])}")

# reversed() — 反转（类比 Java: Collections.reverse(list)）
print(f"reversed([1,2,3]) = {list(reversed([1, 2, 3]))}")

# zip() — 并行遍历（类比 Java: 两个 Iterator 同时 next()）
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(f"  {name} is {age}")

# enumerate() — 带索引遍历（类比 Java: for (int i = 0; i < list.size(); i++)）
for i, name in enumerate(names):
    print(f"  [{i}] {name}")

# print(f"\n{'=' * 50}")
# print("  Python 语法速查完成！")
# print("{'=' * 50}")
print("\n" + "=" * 50)
print("  Python 语法速查完成！")
print("=" * 50)
