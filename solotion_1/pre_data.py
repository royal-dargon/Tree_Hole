# 这个文件是希望能够将数据集提取出来构建一个直接的特征文件，在以后的训练中不需要重复进行提取
import os

from PIL import Image
from transformers import BertTokenizer
import torch


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
    input_ids, masks = [], []
    for r in rows:
        input_id = tokenizer.encode(
            r,
            add_special_tokens=True,
            max_length=100,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        mask = [1 if t != 0 else 0 for t in input_id[0, :].tolist()]
        # mask = torch.tensor(mask).reshape([1, -1])
        input_ids.append(input_id)
        masks.append(mask)
    input_ids = [item.tolist() for item in input_ids]
    input_ids = torch.tensor(input_ids)
    input_ids = torch.squeeze(input_ids, dim=1)
    masks = torch.tensor(masks)
    text_id = [input_ids, masks]
    return text_id


def label2features(labels):
    label_ids = []
    for label in labels:
        label_id = torch.zeros([3])
        # label_id = torch.tensor([label_name.index(label)], dtype=torch.long)
        label_id[label_name.index(label)] = 1
        label_ids.append(label_id)
    label_ids = [item.tolist() for item in label_ids]
    label_ids = torch.tensor(label_ids, dtype=torch.float)
    return label_ids





