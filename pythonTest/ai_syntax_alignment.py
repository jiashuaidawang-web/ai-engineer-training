"""
文件名：ai_syntax_alignment.py
描述：针对 AI 课程高频使用的 Python 特有语法（切片、解包）进行 Java 概念对齐
"""


def demo_split_slicing():
    print("this is demo play python is slicing")

    data_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Java 中裁剪 List 需要：list.subList(start, end)
    # Python 使用 [start:end:step]（左闭右开），极其高效

    # 原始数据集
    print(f"data_list原始数据: {data_list}")

    # 场景 A：获取索引 1 到 4 的元素（左闭右开，不包含 4）
    sub_data = data_list[1:4]
    print(f"获取区间 [1:4]: {sub_data}")

    # 场景 B：省略开头或结尾（代表从头开始，或一直到最后）
    sub_data = data_list[:-1]
    print(f"从开头 到倒数第二个: {sub_data}")


def demo_unpackage():
    print("\n=== 2. 解包 (Unpacking) -> 多返回值与坐标拆解 ===")

    # Java 的方法只能返回一个对象。如果想返回多个值，必须封装成 Map 或自定义 Result 类。
    # Python 的方法可以直接返回多个值（本质是 Tuple），接收时可以直接“拆箱/解包”给多个变量。
    def getMultiValue():
        x = 10
        y = 245
        confidence = 0.5
        return x, y, confidence

    center_x, center_y, center_confidence = getMultiValue();
    print(f"center_x, center_y, center_confidence: {center_x, center_y, center_confidence}")


if __name__ == '__main__':
    demo_split_slicing()
    demo_unpackage()
