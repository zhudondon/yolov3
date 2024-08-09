直接运行会报错
pip install ultralytics==8.1.0
当前环境python 3.12

报错信息
assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"
AssertionError: mydata\weight\yolov3.weights acceptable suffix is
手贱改了文件，应该是.pt结尾的

改回来，现在这个状态，可以进行监测了；接下来就是diy数据

