#1、导入包
import torch
import torch.nn as nn # neural network神经网络包
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import tqdm
from torchnet import meter
from test import generate
 
 
# 2 加载训练数据集
'''实验提供预处理过的数据集，含有57580首唐诗，
每首诗限定在125词，不足125词的以</s>填充。
数据集以npz文件形式保存，包含三个部分：'''
def prepareData():
    datas = np.load("./data/tang.npz")
    data = datas["data"]
    ix2word = datas['ix2word'].item()  # 序号到字的映射
    word2ix = datas['word2ix'].item()  # 字到序号的映射
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data,
                         batch_size=16,
                         shuffle=True,
                         num_workers=2)
    return dataloader, ix2word, word2ix


#3、定义模型：两层卷积
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 网络主要结构 LSTM层 三层双向LSTM 尝试双向的LSTM
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3)#,bidirectional=True)
        # 进行分类
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None: #'''3*2=num_layers* num_directions'''
            h_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(3, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        # 输入 序列长度 * batch(每个汉字是一个数字下标)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        # 输出 序列长度 * batch * 向量维度
        # 输出hidden的大小： 序列长度 * batch * hidden_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden



# 超参数
num_epochs = 20 # 所有数据迭代训练的次数
batch_size = 16 
learning_rate = 1e-3 #学习率

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
 
def train(dataloader, ix2word, word2ix):
    # 定义模型
    print("begin train")
    model = PoetryModel(len(word2ix),
                      embedding_dim=128,
                      hidden_dim=256)
    if torch.cuda.is_available():
        model = model.cuda()
    #4、 选择损失函数和优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()
    # 5、开始训练
    for epoch in range(num_epochs):
        loss_meter.reset()
        for li,data_ in tqdm.tqdm(enumerate(dataloader)):
            #print(data_.shape)
            data_ = data_.long().transpose(1,0).contiguous()
            # 注意这里，也转移到了计算设备上
            data_ = get_variable(data_)
            optimizer.zero_grad()
            # n个句子，前n-1句作为输入，后n-1句作为输出，二者一一对应
            input_,target = data_[:-1,:],data_[1:,:]
            output,_ = model(input_)
            #print("Here",output.shape)
            # 这里为什么view(-1)
            #print(target.shape,target.view(-1).shape)
            loss = criterion(output,target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            # 进行可视化
            if (1+li) % 100 == 0:
                print('Epoch [%d/%d], train Loss is: %s'%(epoch + 1, num_epochs, str(loss_meter.mean)))
                words = u"湖光秋月两相和"
                gen_poetry = ''.join(generate(model,words,ix2word,word2ix))
                print(gen_poetry)
        
        #6、保存训练模型 Save the Trained Model
        torch.save(model.state_dict(),"Poetry.pkl")

if __name__ == '__main__':
    dataloader, ix2word, word2ix  = prepareData()
    train(dataloader, ix2word, word2ix)
