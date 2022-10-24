# 这个文件是希望能够将数据集提取出来构建一个直接的特征文件，在以后的训练中不需要重复进行提取
import os

from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


bert_en_model = "../pre_model/pretrained_berts/bert_en"
tokenizer = BertTokenizer.from_pretrained(bert_en_model)
model = BertModel.from_pretrained(bert_en_model)


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


if __name__ == "__main__":
    # 获取文件夹的源数据
    data = get_single()
    # 将这些row的数据送入BERT与res-net进行特征的提取

