# 1、导入包
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms
 
#transform = transforms.Compose([transforms.ToTensor(),
                                #transforms.Normalize(mean=[0.5],std=[0.5])])
# 2、导入测试数据
test_dataset = datasets.MNIST(root="./data/",
                           transform = transforms.ToTensor(),
                            train = False)
 
test_loader = torch.utils.data.DataLoader(dataset =test_dataset,
                                                batch_size = 100,
                                                shuffle = True)

#3、模型，同训练集模型一样 两层卷积
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

#4、 加载训练好的模型参数
model.load_state_dict(torch.load('cnn1.pkl'))

# 5、在测试集上计算正确率
count = 0
right_count = 0
for i, (images, labels) in enumerate(test_loader):
    count += images.shape[0]
    outputs = model(images)
    predect_label =  torch.max(outputs, 1)[1].data.squeeze()
    right_count += torch.sum(predect_label==labels).item()
    #可视化测试过程
    if i % 10 == 0:
        print("processed [%d/10000] images, now the correctness is %.4f" %(i*100 , right_count / count))

print("end...")
print("the final correctness in test dataset is ", right_count / count)
