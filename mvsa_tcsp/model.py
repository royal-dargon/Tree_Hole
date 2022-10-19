import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from copy import deepcopy


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
                            bidirectional=True,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=config['lstm_hidden_size'] * 2, out_features=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        前反馈神经网络
        :param x: shape （序列长度，batch_size, 特征数）
        """
        h, c = self.init_lstm_para()
        output, (_, _) = self.lstm(x, (h, c))
        # 使用batch_first后输出的变成[batch, length, hidden]
        out = output[:, -1, :].squeeze(0)
        # 准备添加自注意力
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


class Translation(nn.Module):
    def __init__(self, config):
        super(Translation, self).__init__()
        self.config = config
        self.encoder = nn.LSTM(input_size=config["source_size"],
                               hidden_size=config["encoder_hidden_size"],
                               num_layers=config["encoder_num_layers"],
                               dropout=config["encoder_dropout"])
        self.bn = nn.BatchNorm1d(config["encoder_hidden_size"])
        self.decoder = nn.LSTM(input_size=config["decoder_input_size"],
                               hidden_size=config["decoder_hidden_size"],
                               num_layers=config["decoder_num_layers"],
                               dropout=config["decoder_dropout"])
        self.attention_1 = nn.Linear(config["encoder_hidden_size"] * 2, config["decoder_hidden_size"])
        self.attention_2 = nn.Linear(config["decoder_hidden_size"], 1, bias=False)
        self.fc_1 = nn.Linear(config["encoder_hidden_size"] + config["decoder_hidden_size"],
                              config["encoder_hidden_size"] + config["decoder_hidden_size"])
        self.fc_2 = nn.Linear(config["encoder_hidden_size"] + config["decoder_hidden_size"], config["target_size"])

    def get_attention_weight(self, dec_rep, enc_reps, mask):
        # dec_rep (1, enc_n, b, dec_h_d), enc_reps (1, enc_n, b, enc_h_d)
        cat_reps = torch.cat([enc_reps, dec_rep], dim=-1)               # (1, enc_n, enc_h_d + dec_h_d)
        attn_scores = self.attention_2(F.tanh(self.attention_1(cat_reps))).squeeze(3)    # (1, enc_n, b)
        attn_scores = mask * attn_scores
        return torch.softmax(attn_scores, dim=1)                        # (1, enc_n, b)

    def encode(self, source, lengths):
        packed_sequence = pack_padded_sequence(source, lengths.cpu())
        packed_hs, (final_h, _) = self.encoder(packed_sequence)
        enc_hs, _ = pad_packed_sequence(packed_hs)                      # (enc_n, b, enc_h_d)
        return enc_hs

    def decode(self, source, target, enc_hs, mask):
        n_step = len(target)
        enc_n, batch_size, enc_h_d = enc_hs.size()
        dec_h_d = self.config["decoder_hidden_size"]

        # initialize
        dec_h = torch.zeros(1, batch_size, dec_h_d).to(source.device)  # (1, b, dec_h_d)
        dec_c = deepcopy(dec_h)

        dec_rep = dec_h.view(1, 1, batch_size, dec_h_d).expand(1, enc_n, batch_size, dec_h_d)  # (1, enc_n, b, dec_h_d)
        enc_reps = enc_hs.view(1, enc_n, batch_size, enc_h_d)  # (1, enc_n, b, enc_h_d)
        attn_weights = self._get_attn_weight(dec_rep, enc_reps, mask)  # (1, enc_n, b)
        context = attn_weights.unsqueeze(3).expand_as(enc_reps) * enc_reps  # (1, enc_n, b, enc_h_d)
        context = torch.sum(context, dim=1)  # (1, b, enc_h_d)

        dec_in = torch.cat([dec_h, context], dim=2)  # (1, b, enc_h_d+dec_h_d)
        all_attn_weights = torch.empty([n_step, enc_n, batch_size]).to(source.device)  # (dec_n, enc_n, b)
        all_dec_out = torch.empty([n_step, batch_size, self.config["target_size"]]).to(
            source.device)  # (dec_n, b, dec_o_d)

        for i in range(n_step):
            _, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))  # (1, b, dec_h_d)

            dec_rep = dec_h.view(1, 1, batch_size, dec_h_d).expand(1, enc_n, batch_size, dec_h_d)
            attn_weights = self._get_attn_weight(dec_rep, enc_reps, mask)  # (1, b, enc_n)
            all_attn_weights[i] = attn_weights

            context = attn_weights.unsqueeze(3).expand_as(enc_reps) * enc_reps  # (1, enc_n, b, enc_h_d)
            context = torch.sum(context, dim=1)  # (1, b, enc_h_d)
            dec_in = torch.cat([dec_h, context], dim=2)
            all_dec_out[i] = self.fc2(F.relu(self.fc1(dec_in)))  # (1, b, dec_o_d)

        return all_dec_out.permute(1, 0, 2).contiguous(), all_attn_weights.permute(2, 0, 1).contiguous()

    def forward(self, source, target, lengths):
        pass

