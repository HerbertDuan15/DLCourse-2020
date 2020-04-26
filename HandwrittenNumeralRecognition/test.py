import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
 
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],std=[0.5])])
 
 
data_test = datasets.MNIST(root="./data/",
                           transform = transform,
                            train = False)
 
 
data_loader_test = torch.utils.data.DataLoader(dataset =data_test,
                                                batch_size = 100,
                                                shuffle = True)

# 两层卷积
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 使用序列工具快速构建
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out


model = CNN()
# 将所有的模型参数移动到GPU上
if torch.cuda.is_available():
    model.cuda()
print(model)

# 卷积神经网络模型进行模型训练和参数优化的代码
model.load_state_dict(torch.load('cnn1.pkl'))

X_test,y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
print(pred)
_,pred = torch.max(pred,1)
print(pred)
 
print("Predict Label is:",[i for i in pred])
print("Real Label is :",[i for i in y_test])
 
img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1,2,0)
 
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std +mean
cv2.imshow('win',img)
key_pressed=cv2.waitKey(0)