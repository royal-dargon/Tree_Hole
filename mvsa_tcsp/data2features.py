import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import sklearn.preprocessing as sp

bert_en_model = "../pre_model/pretrained_berts/bert_en"
tokenizer = BertTokenizer.from_pretrained(bert_en_model)
model = BertModel.from_pretrained(bert_en_model)
# r = tokenizer.tokenize("From Home Work to Modern Manufacture. Modern manufacturing has changed over time.")


# create the list of word list
class Vocab:
    def __init__(self, vocab_path):
        self.UNK = '[UNK]'
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(self.UNK))

    def __len__(self):
        return len(self.itos)


# test the word list
vocab = Vocab("../pre_model/pretrained_berts/bert_en/vocab.txt")
# print(vocab.stoi['good'])


def text2features(text_rows):
    text_features = []
    for row in text_rows:
        inputs_id = tokenizer.encode(
            row,
            add_special_tokens=True,
            max_length=100,
            padding='max_length',
            return_tensors='pt'
        )
        # print(inputs_id.shape)
        mask = [1 if t != 0 else 0 for t in inputs_id[0, :].tolist()]
        mask = torch.tensor(mask).reshape([1, -1])
        out_put = model(inputs_id, attention_mask=mask)
        out_put = torch.squeeze(out_put[0], 0)
        text_features.append(out_put.detach().numpy())
        # print(out_put[0].shape, out_put[1].shape)
    return np.array(text_features)


class Images2Features(nn.Module):
    def __init__(self):
        super(Images2Features, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def image2features(images_rows):
    net = Images2Features()
    net.eval()
    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    res = []
    for row in images_rows:
        img_t = process(row)
        batch_t = torch.unsqueeze(img_t, 0)
        feat = net(batch_t)
        feat = torch.squeeze(feat, 0)
        res.append(feat.detach().numpy())
    return np.array(res)


label_name = ['neutral', 'negative', 'positive']


def label2features(label_rows):
    res = []
    for label in label_rows:
        label_id = torch.tensor([label_name.index(label)], dtype=torch.long)
        res.append(label_id)
    return res


