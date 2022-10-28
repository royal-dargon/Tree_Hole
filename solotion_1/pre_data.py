# 这个文件是希望能够将数据集提取出来构建一个直接的特征文件，在以后的训练中不需要重复进行提取
import os

from PIL import Image
from transformers import BertTokenizer
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import torch.optim as optim

import model


bert_en_model = "../pre_model/pretrained_berts/bert_en"
tokenizer = BertTokenizer.from_pretrained(bert_en_model)

label_name = ['neutral', 'negative', 'positive']


def get_single():
    data_path = "../data/MVSA_Single/data"
    label_path = "../data/MVSA_Single/labelResultAll.txt"
    file_name = os.listdir(data_path)
    file_name.sort()
    data_rows = {
        "text": [],
        "image": [],
        "text_labels": [],
        "image_labels": []
    }
    labels = {}
    i = 0
    f = open(label_path)
    lines = f.readlines()
    for line in lines:
        if i == 0:
            i += 1
            continue
        line = line.split(sep="\t")
        index = line[0]
        line = line[1].replace('\n', '')
        line = line.split(",")
        labels[index] = [line[0], line[1]]
        i += 1
    for name in file_name:
        index = name.split(".")[0]
        if name.endswith("txt"):
            f = open(data_path + "/" + name, encoding='unicode_escape')
            lines = f.readlines()
            s = ""
            for line in lines:
                line = line.strip("\n")
                s += line
            data_rows["text"].append(s)
        elif name.endswith("jpg"):
            i = Image.open(data_path + "/" + name)
            data_rows["image"].append(i)
        data_rows["text_labels"].append(labels[index][0])
        data_rows["image_labels"].append(labels[index][1])

    return data_rows


def text2id(rows):
    input_id = tokenizer.encode(
        rows,
        add_special_tokens=True,
        max_length=100,
        padding='max_length',
        return_tensors='pt'
    )
    mask = [1 if t != 0 else 0 for t in input_id[0, :].tolist()]
    mask = torch.tensor(mask).reshape([1, -1])
    text_id = [input_id, mask]
    return text_id


def label2features(label):
    label_id = torch.tensor([label_name.index(label)], dtype=torch.long)
    return label_id


if __name__ == "__main__":
    # 获取文件夹的源数据
    data = get_single()
    # 将这些row的数据送入BERT与res-net进行特征的提取
    # text = text2id(data["text"])
    epoch = 10
    TextModel = model.Text2Features()
    text_optim = optim.Adam([param for param in TextModel.parameters()], lr=1e-5)
    loss_func_text = nn.CrossEntropyLoss()
    # 文本模型的测试
    for i in range(epoch):
        # 注意一下标签的序号
        j = 0
        for row in data["text"][:2000]:
            text_optim.zero_grad()
            text = text2id(row)
            out = TextModel(text)
            out = torch.unsqueeze(out, dim=0)
            text_label = data["text_labels"][j]
            j += 1
            y = label2features(text_label)
            loss = loss_func_text(out, y)
            loss.backward()
            text_optim.step()
            if j % 10 == 0:
                print(loss, out)
    # 保存模型的参数
    torch.save(TextModel.state_dict(), './save/single/textmodel.pt')




