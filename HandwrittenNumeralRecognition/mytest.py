#_*_coding:utf-8_*_
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision包的主要功能是实现数据的处理，导入和预览等
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
 
start_time = time.time()
# 对数据进行载入及有相应变换,将Compose看成一种容器，他能对多种数据变换进行组合
# 传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作
#ToTensor:convert a PIL image to tensor (HWC) in range [0,255] to a torch.Tensor(CHW) in the range [0.0,1.0]
#Normalize：Normalized an tensor image with mean and standard deviation
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])
                                ])
 
 
# 首先获取手写数字的训练集和测试集
# root 用于指定数据集在下载之后的存放路径
# transform 用于指定导入数据集需要对数据进行那种变化操作
# train是指定在数据集下载完成后需要载入那部分数据，
# 如果设置为True 则说明载入的是该数据集的训练集部分
# 如果设置为FALSE 则说明载入的是该数据集的测试集部分
data_train = datasets.MNIST(root="./data/",
                           transform = transform,
                            train = True,
                            download = True)
 
data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                            train = False)
 
 
#数据预览和数据装载
# 下面对数据进行装载，我们可以将数据的载入理解为对图片的处理，
# 在处理完成后，我们就需要将这些图片打包好送给我们的模型进行训练 了  而装载就是这个打包的过程
# dataset 参数用于指定我们载入的数据集名称
# batch_size参数设置了每个包中的图片数据个数
#  在装载的过程会将数据随机打乱顺序并进打包
data_loader_train = torch.utils.data.DataLoader(dataset =data_train,
                                                batch_size = 64,
                                                shuffle = True)
data_loader_test = torch.utils.data.DataLoader(dataset =data_test,
                                                batch_size = 64,
                                                shuffle = False)
 
# 在装载完成后，我们可以选取其中一个批次的数据进行预览
print("one batch images previewing.......")
images,labels = next(iter(data_loader_train))
print(images[0])
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)#理解transpose：https://blog.csdn.net/u012762410/article/details/78912667
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std +mean
# print(labels)
print([labels[i] for i in range(64)])
# 由于matplotlab中的展示图片无法显示，所以现在使用OpenCV中显示图片
cv2.imshow('win',img)
print("press any key to continue.....")
key_pressed=cv2.waitKey(0)


class myNet(nn.Module):
    #input: 1*1*28*28
    print("loading myNet module")
    def __init__(self):
        super().__init__()
        #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1,10,4)
        self.fc = nn.Linear(10*24*24,10)


    def forward(self, x):
        out = self.conv(x)#1*10*25*25
        out = F.relu(out)
        out = F.max_pool2d(out,2)# 1* 10 * 24 * 24
        print(out.size(0))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim = 1)
        return out

model = myNet()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

for epoch in range(5):
    for i,(images,labels) in enumerate(data_loader_train):
        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        loss = loss_func(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) %64 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, 5, i + 1, len(data_loader_train) // 64, loss.item()))
 
 
# Save the Trained Model
torch.save(model.state_dict(), 'cnn.pkl')



