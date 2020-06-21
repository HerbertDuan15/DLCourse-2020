# 1、 导入包
# from getdata import DogsVSCatsDataset as DVCD
# from network import Net
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
# import getdata
from torch.utils.data import DataLoader as DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms


#2、导入数据
dataset_dir = './data/test/'                    # 数据集路径
# 默认输入网络的图片大小
IMAGE_SIZE = 200

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                          # 将图像按比例缩放至合适尺寸
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),        # 从图像中心裁剪合适大小的图像
    transforms.ToTensor()   # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
])

class DogsVSCatsDataset(data.Dataset):      # 新建一个数据集类，并且需要继承PyTorch中的data.Dataset父类
    def __init__(self, dir):          # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.list_img = []                  # 新建一个image list，用于存放图片路径，注意是图片路径
        self.list_label = []                # 新建一个label list，用于存放图片对应猫或狗的标签，其中数值0表示猫，1表示狗
        self.data_size = 0                  # 记录数据集大小
        self.transform = dataTransform      # 转换关系

        for file in os.listdir(dir):    # 遍历dir文件夹
            self.list_img.append(dir + file)        # 将图片路径和文件名添加至image list
            self.data_size += 1                     # 数据集增1
            name = file.split(sep='.')              # 分割文件名，"cat.0.jpg"将分割成"cat",".","jpg"3个元素
            # label采用one-hot编码，"1,0"表示猫，"0,1"表示狗，任何情况只有一个位置为"1"，在采用CrossEntropyLoss()计算Loss情况下，label只需要输入"1"的索引，即猫应输入0，狗应输入1
            if name[0] == 'cat':
                self.list_label.append(0)         # 图片为猫，label为0
            else:
                self.list_label.append(1)         # 图片为狗，label为1，注意：list_img和list_label中的内容是一一配对的
                
    def __getitem__(self, item):            # 重载data.Dataset父类方法，获取数据集中数据内容
        img = Image.open(self.list_img[item])                       # 打开图片
        label = self.list_label[item]                               # 获取image对应的label
        return self.transform(img), torch.LongTensor([label])       # 将image和label转换成PyTorch形式并返回

    def __len__(self):
        return self.data_size               # 返回数据集大小

# 3 加载模型
model_file = './cnn2-3cov.pkl'                # 模型保存路径
N = 10
class Net(nn.Module):                                       # 新建一个网络类，就是需要搭建的网络，必须继承PyTorch的nn.Module父类
    def __init__(self):                                     # 构造函数，用于设定网络层
        super(Net, self).__init__()                         # 标准语句
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 第二个卷积层，输入通道数16，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
        self.conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 第三个卷积层，输入通道数16，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
        
        ###############两层卷积时此处线性函数输入为50*50*16，三层卷积时为25*25*16
        self.fc1 = nn.Linear(25*25*16, 128)                 # 第一个全连层，线性连接，输入节点数50×50×16，输出节点数128
        self.fc2 = nn.Linear(128, 64)                       # 第二个全连层，线性连接，输入节点数128，输出节点数64
        self.fc3 = nn.Linear(64, 2)                         # 第三个全连层，线性连接，输入节点数64，输出节点数2

    def forward(self, x):                   # 重写父类forward方法，即前向计算，通过该方法获取网络输入数据后的输出值
        #print(x.shape)
        x = self.conv1(x)                   # 第一次卷积
        x = F.relu(x)                       # 第一次卷积结果经过ReLU激活函数处理
        x = F.max_pool2d(x, 2)              # 第一次池化，池化大小2×2，方式Max pooling

        x = self.conv2(x)                   # 第二次卷积
        x = F.relu(x)                       # 第二次卷积结果经过ReLU激活函数处理
        x = F.max_pool2d(x, 2)              # 第二次池化，池化大小2×2，方式Max pooling

        ##########在两层卷积的基础上加上第三层卷积试一试
        x = self.conv2(x)                   # 第三次卷积
        x = F.relu(x)                       # 第三次卷积结果经过ReLU激活函数处理
        x = F.max_pool2d(x, 2)              # 第三次池化，池化大小2×2，方式Max pooling

        x = x.view(x.size()[0], -1)         # 由于全连层输入的是一维张量，因此需要对输入的[50×50×16]格式数据排列成[40000×1]形式
        x = F.relu(self.fc1(x))             # 第一次全连，ReLU激活
        x = F.relu(self.fc2(x))             # 第二次全连，ReLU激活
        y = self.fc3(x)                     # 第三次激活，ReLU激活
        return y

# 4、 开始测试
batch_size = 16 
workers = 10
# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
 
def test():
    # get data
    datafile = DogsVSCatsDataset(dataset_dir)                                                           # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)     # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果
    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    # setting model
    model = Net()                                       # 实例化一个网络
    if torch.cuda.is_available():
        model = model.cuda()                # 网络送入GPU，即采用GPU计算，如果没有GPU加速，可以去掉".cuda()"
    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(torch.load(model_file,map_location='cpu'))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout


    # calculation
    count = 0
    right_count = 0
    i = 0
    for img, label in dataloader:
        i = i + 1                                           # 循环读取封装后的数据集，其实就是调用了数据集中的__getitem__()方法，只是返回数据格式进行了一次封装
        img, label = get_variable(img), get_variable(label)           # 将数据放置在PyTorch的Variable节点中，并送入GPU中作为网络计算起点
        out = model(img)
        out = F.softmax(out, dim=1)                         # 输出概率化
        #out = out.data.cpu().numpy()                        # 转成numpy数据
        #print(label)
        #print(label.squeeze())
        label = label.squeeze()
        #print(out)
        predect_label =  torch.max(out, 1)[1].data.squeeze()
        #print(predect_label)
        count += img.shape[0]
        right_count += torch.sum(predect_label==label).item()
        #print(right_count)
        #可视化测试过程
        if i % 10 == 0:
            print("processed [%d/5000] images, now the correctness is %.4f" %(i*batch_size , right_count / count))

        #break;

    print("end...")
    print("the final correctness in test dataset is ", right_count / count)  


if __name__ == '__main__':
    test()
