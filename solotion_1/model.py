# the model about the work

import torch.nn as nn
from transformers import BertModel


class Text2Features(nn.Module):
    def __init__(self, config, batch):
        super(Text2Features, self).__init__()
        self.bert = BertModel.from_pretrained("../pre_model/pretrained_berts/bert_en")          # 从存放的路径加载预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True      # 让参数变成可更新

        self.gru = nn.GRU(input_size=768, hidden_size=1024, batch_first=True,
                          dropout=0.5, num_layers=2, bidirectional=True)

    def forward(self, x):
        """
        :param x: [输入的token序列，序列的真实际长度，mask序列]
        :return:
        """
        context = x[0]
        mask = x[2]

        encoder, pooled = self.bert(context, attention_mask=mask, output_all_encoder_layers=False)
        out, h0 = self.gru(encoder)
        out = out[:, -1, :]                 # 只要最后一个token的输出
        return out





