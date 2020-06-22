#1、导入包
import torch
import torch.nn as nn # neural network神经网络包
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import tqdm
from torchnet import meter
 
 
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

        embeds = self.embeddings(input)

        output, hidden = self.lstm(embeds, (h_0, c_0))

        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden


max_gen_len = 200
# 给定首句生成诗歌
def generate(model, start_words, ix2word, word2ix):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    if torch.cuda.is_available():
        input = input.cuda()
    hidden = None
    model.eval()

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_words_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            # print(output.data[0].topk(1))
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results



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
    print(model)
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
            data_ = get_variable(data_)
            optimizer.zero_grad()
            input_,target = data_[:-1,:],data_[1:,:]
            output,_ = model(input_)
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
