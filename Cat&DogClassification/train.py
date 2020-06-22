# 1、导入包
from torch.utils.data import DataLoader as DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn

import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

import torch.utils.data
import torch.nn.functional as F

# 2、导入数据
dataset_dir = './data/train/'  # 数据集路径
# 默认输入网络的图片大小
IMAGE_SIZE = 200
dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),          # 缩放
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),      # 裁剪
    transforms.ToTensor()   # 归一化
])

class DogsVSCatsDataset(data.Dataset):      # 处理数据集类
    def __init__(self, dir):          
        self.list_img = []                  # 存放图片路径
        self.list_label = []                # 存放图片的标签
        self.data_size = 0                  # 数据集大小
        self.transform = dataTransform      # 转换关系

        for file in os.listdir(dir):    
            self.list_img.append(dir + file)    
            self.data_size += 1                 
            name = file.split(sep='.')       # 分割文件名
            if name[0] == 'cat':
                self.list_label.append(0)         # 图片为猫，label为0
            else:
                self.list_label.append(1)         # 图片为狗，label为1

    def __getitem__(self, item):   
        img = Image.open(self.list_img[item])                     
        label = self.list_label[item]       
        return self.transform(img), torch.LongTensor([label])  

    def __len__(self):
        return self.data_size   # 返回数据集大小


# 3、定义网络
class Net(nn.Module):         
    def __init__(self):                                    
        super(Net, self).__init__()                         
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)  
        self.conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)  
        
        ###############两层卷积时此处线性函数输入为50*50*16，三层卷积时为25*25*16
        self.fc1 = nn.Linear(25*25*16, 128)                
        self.fc2 = nn.Linear(128, 64)                   
        self.fc3 = nn.Linear(64, 2)                     

    def forward(self, x):                  
        #print(x.shape)
        x = self.conv1(x)                  
        x = F.relu(x)                      
        x = F.max_pool2d(x, 2)           

        x = self.conv2(x)               
        x = F.relu(x)                  
        x = F.max_pool2d(x, 2)       

        ##########在两层卷积的基础上加上第三层卷积试一试
        x = self.conv2(x)                   
        x = F.relu(x)                     
        x = F.max_pool2d(x, 2)             

        x = x.view(x.size()[0], -1)         # 由于全连层输入的是一维张量，因此需要对输入的[50×50×16]格式数据排列成[40000×1]形式
        x = F.relu(self.fc1(x))             
        x = F.relu(self.fc2(x))            
        y = self.fc3(x)                    
        return y


# 4、 开始训练
workers = 10                        # PyTorch读取数据线程数量
batch_size = 16                     # batch_size大小
lr = 0.0001                         # 学习率
nepoch = 10

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
 
def train():
    datafile = DogsVSCatsDataset(dataset_dir)                                                           # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)     # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = Net()                       # 实例化一个网络
    print(model)
    if torch.cuda.is_available():
        model = model.cuda()               
    model = nn.DataParallel(model)
    model.train()                       # 网络设定为训练模式，采用了dropout策略，可以放置网络过拟合

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   
    criterion = torch.nn.CrossEntropyLoss()                

    cnt = 0            
    for epoch in range(nepoch):
        for img, label in dataloader:                                           
            img, label = get_variable(img), get_variable(label)          
            out = model(img)                                               
            loss = criterion(out, label.squeeze())    
            loss.backward()                          
            optimizer.step()                        
            optimizer.zero_grad()                   
            cnt += 1

            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))      
    # 5、 保存模型
    torch.save(model.state_dict(), 'cnn2.pkl')            # 训练所有数据后，保存网络的参数


if __name__ == '__main__':
    train()