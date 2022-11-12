# the model about the work
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import logging
import torchvision

text_hidden_size = 1024
text_hidden_size_2 = 1024

logging.set_verbosity_warning()


class Text2Features(nn.Module):
    def __init__(self, device, use_gpu):
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
        self.softmax = nn.Softmax(dim=2)

        self.gru_2 = nn.GRU(input_size=text_hidden_size*2, hidden_size=text_hidden_size_2, batch_first=True,
                            dropout=0.5, num_layers=2, bidirectional=False)
        # 多层全连接
        self.linear_1 = nn.Linear(in_features=text_hidden_size_2, out_features=512)
        self.linear_2 = nn.Linear(in_features=512, out_features=256)
        self.linear_3 = nn.Linear(in_features=256, out_features=3)
        self.tanh = nn.Tanh()
        self.softmax_mlp = nn.Softmax(dim=1)
        self.device = device
        self.use_gpu = use_gpu

    def forward(self, x):
        """
        :param x: [输入的token序列，mask序列]
        :return:
        """
        if self.use_gpu:
            context = x[0].to(self.device)
            mask = x[1].to(self.device)
        else:
            context = x[0]
            mask = x[1]

        encoder, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        # encoder = torch.squeeze(encoder[0], 0)
        out, _ = self.gru(encoder)      # (batch, len, hidden*2)
        q = self.Q_linear(out)
        k = self.K_linear(out)
        v = self.K_linear(out)

        k = k.permute(0, 2, 1)
        alpha = torch.matmul(q, k)  # (batch, 100, 100)
        alpha = self.softmax(alpha)
        out = torch.matmul(alpha, v)  # (batch, 100, 2048)
        _, h0 = self.gru_2(out)
        h0 = h0.permute(1, 0, 2)
        out = torch.squeeze(h0[:, -1, :], dim=1)
        out = self.linear_1(out)
        out = self.tanh(out)
        out = self.linear_2(out)
        out = self.tanh(out)
        out = self.linear_3(out)
        out = self.tanh(out)
        out = self.softmax_mlp(out)

        return out


class Image2Features(nn.Module):
    def __init__(self, is_gpu=True):
        super(Image2Features, self).__init__()
        self.is_gpu = is_gpu

    def forward(self, x):
        pass


class MultiModel(nn.Module):
    def __init__(self, is_gpu=True):
        super(MultiModel, self).__init__()
        self.is_gpu = is_gpu
        self.bert = BertModel.from_pretrained("../pre_model/pretrained_berts/bert_cn")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(input_size=768, hidden_size=text_hidden_size,
                            batch_first=True, num_layers=2, bidirectional=True)
        self.image_model = torchvision.models.resnet152(pretrained=True)

        def save_output(module, inputs, output):
            self.buffer = output
        self.image_model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        if self.use_gpu:
            context = x[0].to(self.device)
            mask = x[1].to(self.device)
        else:
            context = x[0]
            mask = x[1]
