# 1、 导入包
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from torch.utils.data import DataLoader as DataLoader
import torch.utils.data as data
import torchvision.transforms as transforms


#2、导入数据
dataset_dir = './data/test/'                    # 数据集路径
# 默认输入网络的图片大小
IMAGE_SIZE = 200


dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                     
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),    
    transforms.ToTensor()  
])

class DogsVSCatsDataset(data.Dataset):   
    def __init__(self, dir):       
        self.list_img = []         
        self.list_label = []       
        self.data_size = 0          
        self.transform = dataTransform  

        for file in os.listdir(dir):   
            self.list_img.append(dir + file)    
            self.data_size += 1                 
            name = file.split(sep='.')        
            if name[0] == 'cat':
                self.list_label.append(0)     
            else:
                self.list_label.append(1)       
                
    def __getitem__(self, item):           
        img = Image.open(self.list_img[item])             
        label = self.list_label[item]                     
        return self.transform(img), torch.LongTensor([label])   

    def __len__(self):
        return self.data_size        

# 3 加载模型
model_file = './cnn2-3cov.pkl'       
N = 10
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

        x = x.view(x.size()[0], -1)      
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))       
        y = self.fc3(x)               
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
    model = Net()                              
    if torch.cuda.is_available():
        model = model.cuda()           
    model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_file))
    else:
        model.load_state_dict(torch.load(model_file,map_location='cpu'))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    count = 0
    right_count = 0
    i = 0
    for img, label in dataloader:
        i = i + 1                                       
        img, label = get_variable(img), get_variable(label)     
        out = model(img)
        out = F.softmax(out, dim=1) 
        label = label.squeeze()
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
