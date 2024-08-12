直接运行会报错
pip install ultralytics==8.1.0
当前环境python 3.12

报错信息
assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"
AssertionError: mydata\weight\yolov3.weights acceptable suffix is
手贱改了文件，应该是.pt结尾的

改回来，现在这个状态，可以进行监测了；接下来就是diy数据



加入自己的数据集
下载voc_label转换的代码 https://pjreddie.com/media/files/voc_label.py
可以不用，用下面那个create_voc_file

运行该文件，我们的目录下会生成三个txt文件2007_train.txt,2007_val.txt,2007_test.txt，VOCdevkit下的VOC2007也会多生成一个labels文件夹，
下面是真正会使用到的label，点开看发现已经转化成YOLOV3需要的格式了。这时候自己的数据集正式完成。

更换分类，更换文件
# train glass和mouth
classes = ["glass","mouth"]

# 必须手写 或者读取文件
classes = ['glass', 'mouth']
create_voc_file.py
调用脚本，把voc格式转成yolo格式，会创建
images 的结构

生成label txt文件 yolo格式


修改配置文件
voc.yaml



错误：
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous
you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute

解决
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


训练提示
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv3  runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs\train', view at http://localhost:6006/

models.common.Conv



1.中断恢复学习
pretrained，resume

 # 预训练权重 通过这个来判断是否预训练，就是说是否存在 权重文件
    pretrained = weights.endswith(".pt")

     # 重新开始 恢复训练 改为true即可
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")

Transferred 71/71 items from runs\train\exp3\weights\last.pt
Resuming training from runs\train\exp3\weights\last.pt from epoch 2 to 100 total epochs




2.迁移学习，冻结层数
freeze

   # 冻结 层数，自己训练自己的参数
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")

    可以看下 yolo原网络的层数 ，新增自己的网络层 因为这里没有新增网络层，所以就不进行测试了，也可以在原有基础上，减少层数训练


3.结合tensorboard
tensorboard --logdir=D:\Users\86159\PycharmProjects\yolov3\runs
