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
        self.W = nn.Linear(self.k, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs):
        in_put = inputs                # 输入实际上是两部分，一个是input，另一个是mask

        # 开始计算score
        h = self.w_layer(in_put)
        print(x.shape, "x.shape")
        h = self.tanh(h)
        score = self.W(h)
        score = self.dropout_layer(score)   # 可以要也可以不要
        print(score.shape, "score")
        # 下面softmax
        score = self.softmax(score)
        print(score, "softmax score")

        # 加权求和
        output = torch.mm(inputs.T, score)
        output /= self.k ** 0.5
        output = torch.squeeze(output)
        return output


# 下面开始设计图像与句向量之间的自注意力层
class ImageTextAttention(nn.Module):
    # 该层的输入有三个部分分别是img_emb, seq_emp, mask
    # img_emb [M, 4096]
    # seq_emb [L, k]
    # mask [L]
    # input [M, K] m个图像对应的文本表示
    def __init__(self, dropout_rate=0.0, input_shape=None):
        super(ImageTextAttention, self).__init__()
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.i = input_shape[0]                 # 表示句向量的个数
        self.k = input_shape[1]                 # 表示句向量的维度
        self.text_layer = nn.Linear(self.k, 1, bias=True)
        self.text_tanh = nn.Tanh()
        self.m = input_shape[2]                 # 表示输入图片的个数
        self.j = input_shape[3]                 # 表示输入图片的维度
        self.image_layer = nn.Linear(self.j, 1, bias=True)
        self.image_tanh = nn.Tanh()
        self.V = nn.Linear(self.i, self.i, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        img_emb, seq_emb = inputs
        q = self.text_layer(seq_emb)
        q = self.text_tanh(q)
        p = self.image_layer(img_emb)
        p = self.text_tanh(p)
        # 内积+映射
        emb = torch.mm(p, q.T)                  # 输出是[M, L]
        emb = emb + q.T                         # 保证是一个[M, L]与[1, L]
        emb = self.V(emb)
        score = self.dropout_layer(emb)
        # 现在还是没有考虑到mask的问题
        score = self.softmax(score)
        # 向量加权求和
        output = torch.mm(score, seq_emb)       # [M, L],[L, K] => [M, K]
        output /= self.k**0.5                   # 归一化
        return output


# 下面开始整体的模型搭建


print("test model1")
model1 = SelfAttention(dropout_rate=0.0, input_shape=2)
x = torch.randn(3, 2)
res = model1(x)
print(res, "res")
print("test model2")
model2 = ImageTextAttention(dropout_rate=0.0, input_shape=[2, 2, 3, 2])
img_test = torch.randn(3, 2)
text_test = torch.randn(2, 2)
res = model2([img_test, text_test])
print(res)

