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


# train、test、val数据集划分(按照batch_size进行划分)
def divide_data(row_data, batch_size, data_length):
    r_data = {
        "train": dict(),
        "test": dict(),
        "val":  dict()
    }

    # train data 0.7    3408
    train_text, train_image, train_text_label, train_image_label = [], [], [], []
    for index in range(0, int(data_length * 0.7), batch_size):
        if index + batch_size >= int(data_length * 0.7):
            word = row_data["text"][index:int(data_length * 0.7)]
            image = row_data["image"][index:int(data_length * 0.7)]
            word_label = row_data["text_labels"][index:int(data_length * 0.7)]
            image_label = row_data["image_labels"][index:int(data_length * 0.7)]
            train_text.append(word)
            train_image.append(image)
            train_text_label.append(word_label)
            train_text_label.append(image_label)
        else:
            train_text.append(row_data["text"][index:index+batch_size])
            train_image.append(row_data["image"][index:index + batch_size])
            train_text_label.append(row_data["text_labels"][index:index + batch_size])
            train_image_label.append(row_data["image_labels"][index:index + batch_size])
    r_data["train"]["text"] = train_text
    r_data["train"]["image"] = train_image
    r_data["train"]["text_label"] = train_text_label
    r_data["train"]["image_label"] = train_image_label

    # test data 0.2     974
    test_text, test_image, test_t_label, test_i_label = [], [], [], []
    for index in range(int(data_length * 0.7), int(data_length * 0.9), batch_size):
        if index + batch_size >= int(data_length * 0.9):
            test_text.append(row_data["text"][index:int(data_length * 0.9)])
            test_image.append(row_data["image"][index:int(data_length * 0.9)])
            test_t_label.append(row_data["text_labels"][index:int(data_length * 0.9)])
            test_i_label.append(row_data["image_labels"][index:int(data_length * 0.9)])
        else:
            test_text.append(row_data["text"][index:index + batch_size])
            test_image.append(row_data["image"][index:index + batch_size])
            test_t_label.append(row_data["text_labels"][index:index + batch_size])
            test_i_label.append(row_data["image_labels"][index:index + batch_size])
    r_data["test"]["text"] = test_text
    r_data["test"]["image"] = test_image
    r_data["test"]["text_label"] = test_t_label
    r_data["test"]["image_label"] = test_i_label

    # val data 0.1      487
    val_text, val_image, val_text_label, val_image_label = [], [], [], []
    for index in range(int(data_length * 0.9), data_length, batch_size):
        if index + batch_size >= data_length:
            val_text.append(row_data["text"][index:data_length])
            val_image.append(row_data["image"][index:data_length])
            val_text_label.append(row_data["text_labels"][index:data_length])
            val_image_label.append(row_data["image_labels"][index:data_length])
        else:
            val_text.append(row_data["text"][index:index + batch_size])
            val_image.append(row_data["image"][index:index + batch_size])
            val_text_label.append(row_data["text_labels"][index:index + batch_size])
            val_image_label.append(row_data["image_labels"][index:index + batch_size])
    r_data["val"]["text"] = val_text
    r_data["val"]["image"] = val_image
    r_data["val"]["text_label"] = val_text_label
    r_data["val"]["image_label"] = val_image_label
    return r_data


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
    data_len = len(data["text"])
    print(data_len)
    res_data = divide_data(data, batch_size=24, data_length=data_len)
    # temp = 0
    # for da in res_data["val"]["text"]:
    #     # print(len(da))
    #     temp += len(da)
    # print(temp)
    # 将这些row的数据送入BERT与res-net进行特征的提取
    epoch = 1
    TextModel = model.Text2Features()
    text_optim = optim.Adam([param for param in TextModel.parameters()], lr=1e-5)
    loss_func_text = nn.CrossEntropyLoss()
    # 文本模型的测试
    for i in range(epoch):
        # 注意一下标签的序号
        j = 0
        for row in data["text"][:10]:
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




