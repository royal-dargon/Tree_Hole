# 这个文件是希望能够将数据集提取出来构建一个直接的特征文件，在以后的训练中不需要重复进行提取
import os

from PIL import Image
from transformers import BertTokenizer
import torch
from torchvision import transforms
import h5py


bert_en_model = "../pre_model/pretrained_berts/bert_en"
tokenizer = BertTokenizer.from_pretrained(bert_en_model)

label_name = ['neutral', 'negative', 'positive']


def get_single():
    """对于这个数据集，我们的选择是只选择两个模态相同结果的数据进行提取"""
    process = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    data_path = "../data/MVSA_Single/data"
    label_path = "../data/MVSA_Single/labelResultAll.txt"
    file_name = os.listdir(data_path)
    file_name.sort()
    data_rows = {
        "text": [],
        "image": [],
        "text_labels": [],
        "image_labels": [],
        "multi_labels": []
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
        if labels[index][0] == 'positive' and labels[index][1] == 'negative':
            continue
        if labels[index][0] == 'negative' and labels[index][1] == 'positive':
            continue
        if name.endswith("txt"):
            f = open(data_path + "/" + name, encoding='unicode_escape')
            lines = f.readlines()
            s = ""
            for line in lines:
                line = line.strip("\n")
                s += line
            data_rows["text"].append(s)
            data_rows["text_labels"].append(labels[index][0])
        elif name.endswith("jpg"):
            i = Image.open(data_path + "/" + name)
            i = process(i)
            i = i.tolist()
            data_rows["image"].append(i)
            data_rows["image_labels"].append(labels[index][1])
    length = len(data_rows["text_labels"])
    print(length)
    for i in range(length):
        if data_rows["image_labels"][i] == data_rows["text_labels"][i]:
            data_rows["multi_labels"].append(data_rows["text_labels"][i])
        elif data_rows["image_labels"][i] == "neutral":
            data_rows["multi_labels"].append(data_rows["text_labels"][i])
        else:
            data_rows["multi_labels"].append(data_rows["image_labels"][i])
    return data_rows


# train、test、val数据集划分(按照batch_size进行划分)
def divide_data(row_data, batch_size, data_length):
    r_data = {
        "train": dict(),
        "test": dict(),
        "val":  dict()
    }

    # train data 0.7    3408
    train_text, train_image, train_text_label, train_image_label, train_multi_label = [], [], [], [], []
    for index in range(0, int(data_length * 0.7), batch_size):
        if index + batch_size >= int(data_length * 0.7):
            word_id = row_data["text"][0][index:int(data_length * 0.7)]
            word_mask = row_data["text"][1][index:int(data_length * 0.7)]
            image = row_data["image"][index:int(data_length * 0.7)]
            word_label = row_data["text_labels"][index:int(data_length * 0.7)]
            image_label = row_data["image_labels"][index:int(data_length * 0.7)]
            multi_label = row_data["multi_labels"][index:int(data_length * 0.7)]
            word = [word_id, word_mask]
            train_text.append(word)
            train_image.append(image)
            train_text_label.append(word_label)
            train_text_label.append(image_label)
            train_multi_label.append(multi_label)
        else:
            train_text.append([row_data["text"][0][index:index+batch_size],
                               row_data["text"][1][index:index+batch_size]])
            train_image.append(row_data["image"][index:index + batch_size])
            train_text_label.append(row_data["text_labels"][index:index + batch_size])
            train_image_label.append(row_data["image_labels"][index:index + batch_size])
            train_multi_label.append(row_data["multi_labels"][index:index + batch_size])
    r_data["train"]["text"] = train_text
    r_data["train"]["image"] = train_image
    r_data["train"]["text_label"] = train_text_label
    r_data["train"]["image_label"] = train_image_label
    r_data["train"]["multi_label"] = train_multi_label

    # test data 0.2     974
    test_text, test_image, test_t_label, test_i_label, test_m_label = [], [], [], [], []
    for index in range(int(data_length * 0.7), int(data_length * 0.9), batch_size):
        if index + batch_size >= int(data_length * 0.9):
            test_text.append([row_data["text"][0][index:int(data_length * 0.9)],
                            row_data["text"][1][index:int(data_length * 0.9)]])
            test_image.append(row_data["image"][index:int(data_length * 0.9)])
            test_t_label.append(row_data["text_labels"][index:int(data_length * 0.9)])
            test_i_label.append(row_data["image_labels"][index:int(data_length * 0.9)])
            test_m_label.append(row_data["multi_labels"][index:int(data_length * 0.9)])
        else:
            test_text.append([row_data["text"][0][index:index + batch_size],
                              row_data["text"][1][index:index + batch_size]])
            test_image.append(row_data["image"][index:index + batch_size])
            test_t_label.append(row_data["text_labels"][index:index + batch_size])
            test_i_label.append(row_data["image_labels"][index:index + batch_size])
            test_m_label.append(row_data["multi_labels"][index:index + batch_size])
    r_data["test"]["text"] = test_text
    r_data["test"]["image"] = test_image
    r_data["test"]["text_label"] = test_t_label
    r_data["test"]["image_label"] = test_i_label
    r_data["test"]["multi_label"] = test_m_label

    # val data 0.1      487
    val_text, val_image, val_text_label, val_image_label, val_multi_label = [], [], [], [], []
    for index in range(int(data_length * 0.9), data_length, batch_size):
        if index + batch_size >= data_length:
            val_text.append([row_data["text"][0][index:data_length],
                             row_data["text"][1][index:data_length]])
            val_image.append(row_data["image"][index:data_length])
            val_text_label.append(row_data["text_labels"][index:data_length])
            val_image_label.append(row_data["image_labels"][index:data_length])
            val_multi_label.append(row_data["multi_labels"][index:data_length])
        else:
            val_text.append([row_data["text"][0][index:index + batch_size],
                             row_data["text"][1][index:index + batch_size]])
            val_image.append(row_data["image"][index:index + batch_size])
            val_text_label.append(row_data["text_labels"][index:index + batch_size])
            val_image_label.append(row_data["image_labels"][index:index + batch_size])
            val_multi_label.append(row_data["multi_labels"][index:index + batch_size])
    r_data["val"]["text"] = val_text
    r_data["val"]["image"] = val_image
    r_data["val"]["text_label"] = val_text_label
    r_data["val"]["image_label"] = val_image_label
    r_data["val"]["multi_label"] = val_multi_label
    return r_data


def text2id(rows):
    input_ids, masks = [], []
    for r in rows:
        input_id = tokenizer.encode(
            r,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
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
    # label_ids = torch.tensor(label_ids, dtype=torch.float)
    return label_ids


def load_data(filepath, batch_size):
    """加载数据:hdf5格式"""
    file = h5py.File(filepath, "r")
    data_rows = {
        "text": [],
        "image": [],
        "text_labels": [],
        "image_labels": [],
        "multi_labels": []
    }
    data_rows["text"].append(file["text"]["text_id"][:])
    data_rows["text"].append(file["text"]["text_mask"][:])
    data_rows["image"] = file["vision"]["image"][:]
    data_rows["text_labels"] = file["labels"]["text_label"][:]
    data_rows["image_labels"] = file["labels"]["image_label"][:]
    data_rows["multi_labels"] = file["labels"]["multi_label"][:]
    length = len(data_rows["image_labels"])
    data = divide_data(data_rows, batch_size, length)

    return data







