直接运行会报错
pip install ultralytics==8.1.0
当前环境python 3.12

报错信息
assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"
AssertionError: mydata\weight\yolov3.weights acceptable suffix is
手贱改了文件，应该是.pt结尾的

改回来，现在这个状态，可以进行监测了；接下来就是diy数据



加入自己的数据集
下载voc_label转换的代码
https://pjreddie.com/media/files/voc_label.py

运行该文件，我们的目录下会生成三个txt文件2007_train.txt,2007_val.txt,2007_test.txt，VOCdevkit下的VOC2007也会多生成一个labels文件夹，
下面是真正会使用到的label，点开看发现已经转化成YOLOV3需要的格式了。这时候自己的数据集正式完成。

更换分类，更换文件
# train glass和mouth
classes = ["glass","mouth"]


create_voc_file.py
调用脚本，把voc格式转成yolo格式，会创建 2007_train.cache 2007_train.cache.npy
images 的结构


修改配置文件
voc.yaml




