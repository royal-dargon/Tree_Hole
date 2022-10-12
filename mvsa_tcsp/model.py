import torch.nn as nn
import torch


class TextModel(nn.Module):
    def __init__(self, config, batch):
        super(TextModel, self).__init__()
        self.config = config
        self.batch_size = batch
        # lstm的输入可以看成每个单词词向量的大小, 这里是使用了双向的lstm
        self.lstm = nn.LSTM(input_size=config['source_size'],
                            hidden_size=config['lstm_hidden_size'],
                            num_layers=config['lstm_num_layers'],
                            dropout=config['lstm_dropout'],
                            bidirectional=True)
        self.fc1 = nn.Linear(in_features=config['lstm_hidden_size'], out_features=3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h, c):
        """
        前反馈神经网络
        :param x: shape （序列长度，batch_size, 特征数）
        """
        output, (hidden, cell_state) = self.lstm(x, (h, c))
        out = hidden[-1]
        out = self.fc1(out)
        out = self.softmax(out)
        return out

    def init_lstm_para(self):
        # 注意第一个参数如果是双向则是需要*2
        h0, c0 = torch.randn(self.config['lstm_num_layers'] * 2, self.batch_size, self.config['lstm_hidden_size']), \
                 torch.randn(self.config['lstm_num_layers'] * 2, self.batch_size, self.config['lstm_hidden_size'])
        return h0, c0
    
    
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()

