# the model about the work
import torch
import torch.nn as nn
from transformers import BertModel

text_hidden_size = 1024
text_hidden_size_2 = 1024


class Text2Features(nn.Module):
    def __init__(self):
        super(Text2Features, self).__init__()
        self.bert = BertModel.from_pretrained("../pre_model/pretrained_berts/bert_en")          # 从存放的路径加载预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True      # 让参数变成可更新

        self.gru = nn.GRU(input_size=768, hidden_size=text_hidden_size, batch_first=True,
                          dropout=0.5, num_layers=2, bidirectional=True)
        # 自注意力 Q(query),K(key),V(Value)
        self.Q_linear = nn.Linear(in_features=text_hidden_size*2, out_features=text_hidden_size*2, bias=False)
        self.K_linear = nn.Linear(in_features=text_hidden_size*2, out_features=text_hidden_size*2, bias=False)
        self.V_linear = nn.Linear(in_features=text_hidden_size*2, out_features=text_hidden_size*2, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.gru_2 = nn.GRU(input_size=text_hidden_size*2, hidden_size=text_hidden_size_2, batch_first=True,
                            dropout=0.5, num_layers=2, bidirectional=False)
        # 多层全连接
        self.linear_1 = nn.Linear(in_features=text_hidden_size_2, out_features=3)
        self.softmax_mlp = nn.Softmax(dim=0)

    def forward(self, x):
        """
        :param x: [输入的token序列，mask序列]
        :return:
        """
        context = x[0]
        mask = x[1]

        encoder, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        encoder = torch.squeeze(encoder[0], 0)
        out, _ = self.gru(encoder)
        q = self.Q_linear(out)
        k = self.K_linear(out)
        v = self.K_linear(out)

        alpha = torch.matmul(q, k.T)    # (100, 100)
        alpha = self.softmax(alpha)
        out = torch.matmul(alpha, v)    # (100, 2048)
        out = torch.unsqueeze(out, dim=0)

        _, h0 = self.gru_2(out)
        out = torch.squeeze(h0[-1, :, :], dim=0)
        out = self.linear_1(out)
        out = self.softmax_mlp(out)

        return out





