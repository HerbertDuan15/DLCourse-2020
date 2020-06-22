#1、导入包
import torch
from train import *

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
        # 如果在给定的句首中，input为句首中的下一个字
        # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
        # 最后的hidden
        if i < start_words_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        # 否则将output作为下一个input进行
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

if __name__ == "__main__":
    words = u"湖光秋月两相和" # 修改此处，完成引子
    _, ix2word, word2ix = prepareData()
    model = PoetryModel(len(word2ix),
                      embedding_dim=128,
                      hidden_dim=256)
    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load('Poetry-20.pkl'))
    else:
        model.load_state_dict(torch.load('Poetry-20.pkl',map_location='cpu'))       # 加载训练好的模型参数
    print('引子：%s'%( words))
    print("根据上面这句话生成诗.....")
    gen_poetry = ''.join(generate(model, words, ix2word, word2ix))
    # gen_poetry = gen_poetry.split(u"。")
    print("生成的诗句如下：\n%s" % (gen_poetry))
