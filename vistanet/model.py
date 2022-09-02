import torch.nn as nn
import torch


# 自注意力层
class SelfAttention(nn.Module):
    # input: [None, n, k] 输入为n个维度为k的词向量
    # mask: [None, n] 表示填充词位置的mask
    # output: [None, k] 输出n个词向量的加权和
    def __init__(self, dropout_rate=0.0, input_shape=None):
        super(SelfAttention, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.k = input_shape        # 词向量的维度
        self.w_layer = nn.Linear(self.k, self.k, bias=True)
        self.tanh = nn.Tanh()
        self.weights = torch.randn(self.k, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        in_put = inputs                # 输入实际上是两部分，一个是input，另一个是mask

        # 开始计算score
        x = self.w_layer(in_put)
        print(x.shape, "x.shape")
        x = self.tanh(x)
        print(self.weights, "weights")
        score = torch.mm(x, self.weights)
        score = self.dropout_layer(score)   # 可以要也可以不要
        print(score, "score")
        # 下面softmax
        score = self.softmax(score)

        # 加权求和
        output = torch.mm(inputs.T, score)
        output /= self.k ** 0.5
        output = torch.squeeze(output)
        return output


model = SelfAttention(dropout_rate=0.0, input_shape=2)
x = torch.randn(3, 2)
res = model(x)
print(res, "res")

