# the model about the work
import math

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import logging
import torchvision

max_length = 128
text_hidden_size = 1024
text_hidden_size_2 = 1024       # text model use

# multi_model
multi_text_hidden_size_1 = 128
img_hidden_size = 2048
img_hidden_size_1 = 512

multi_size = multi_text_hidden_size_1 * 2
hidden_size = 128


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


# hyper parameters
multi_max_length = 128
multi_text_hidden_size_lstm = 98
multi_transform_size_1 = 196


class MultiModel(nn.Module):
    def __init__(self, device, is_gpu):
        super(MultiModel, self).__init__()
        self.bert = BertModel.from_pretrained("../pre_model/pretrained_berts/bert_en")  # 从存放的路径加载预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True  # 让参数变成可更新
        self.lstm = nn.LSTM(input_size=768, hidden_size=text_hidden_size, dropout=0.5,
                            batch_first=True, num_layers=2, bidirectional=True)
        self.image_model = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')

        def save_output(module, inputs, output):
            self.buffer = output
        self.image_model.layer4.register_forward_hook(save_output)

        # self.linear_2 = nn.Linear(in_features=2048 * 7 * 7, out_features=2048)
        self.average = nn.AdaptiveAvgPool1d(1)
        self.fusion_layer = nn.Transformer(d_model=2048, batch_first=True,
                                           num_encoder_layers=1, num_decoder_layers=1)

        self.reg_layer = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=3),
        )

        self.device = device
        self.is_gpu = is_gpu

    def forward(self, x_t, x_image):
        if self.is_gpu:
            contexts = x_t[0].to(self.device)
            masks = x_t[1].to(self.device)
            image = x_image.to(self.device)
        else:
            contexts = x_t[0]
            masks = x_t[1]
            image = x_image

        encoder, pooled = self.bert(contexts, attention_mask=masks, return_dict=False)
        out, (_, _) = self.lstm(encoder)                        # (batch_size, length, hidden_size * 2)
        out = out[:, -1, :]                                     # 只是选择最后一个进行输出 (64, 2048)
        out = torch.unsqueeze(out, dim=1)
        _ = self.image_model(image)                             # (batch_size, 2048, 7, 7)
        img = self.buffer
        img = img.view(img.size(0), 2048, -1)
        img = self.average(img)
        img = img.permute(0, 2, 1)
        # img = img.view(img.size(0), -1)        # tree dims to 1
        # img = self.linear_2(img)
        # img = torch.unsqueeze(img, dim=1)
        out = self.fusion_layer(out, img)
        out = torch.squeeze(out, dim=1)
        out = self.reg_layer(out)
        return out


