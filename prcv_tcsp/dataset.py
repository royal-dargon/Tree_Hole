import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from transformers import BertTokenizer
import numpy as np


label_name = ['neutral', 'negative', 'positive']
bert_en_model = "../pre_model/pretrained_berts/bert_en"
tokenizer = BertTokenizer.from_pretrained(bert_en_model)


process = T.Compose([
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


class MVSADataset(Dataset):
    def __init__(self, data_dir, label_path, dataset='single', transform=None, mood="train"):
        self.data_dir = data_dir
        self.label_path = label_path
        self.dataset = dataset
        self.transform = transform
        if dataset == 'single':
            pass
        else:
            self.data_info = self.get_multi_data_info(data_dir, label_path, mood)

    def __len__(self):
        return len(self.data_info["texts_info"])

    def __getitem__(self, index):
        path_text, path_img, text_label, img_label, multi_label = self.data_info["texts_info"][index][0], \
                                                                  self.data_info["images_info"][index][0], \
                                                                  self.data_info["texts_info"][index][1], \
                                                                  self.data_info["images_info"][index][1], \
                                                                  self.data_info["labels"][index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        f = open(path_text, encoding='unicode_escape')
        lines = f.readlines()
        s = ""
        for line in lines:
            line = line.strip("\n")
            s += line

        input_id = tokenizer.encode(
            s,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_id = torch.squeeze(input_id)
        mask = [1 if t != 0 else 0 for t in input_id[:].tolist()]
        mask = torch.tensor(mask)
        data = {"text_id": input_id, "text_mask": mask, "img": img, "label": multi_label}
        return data

    @staticmethod
    def get_multi_data_info(data_dir, label_path, mood):
        data_info = {
            "images_info": list(),
            "texts_info": list(),
            "labels": list()
        }

        # load labels
        labels, i = {}, 0
        f = open(label_path)
        lines = f.readlines()
        for line in lines:
            if i == 0:
                i += 1
                continue
            line = line.split(sep="\t")
            index = line[0]
            line[3] = line[3].replace('\n', '')
            row_1, row_2, row_3 = line[1].split(','), line[2].split(','), line[3].split(',')
            # judgment label
            if row_1[0] == row_2[0] or row_1[0] == row_3[0]:
                text_label = row_1[0]
            elif row_2[0] == row_3[0]:
                text_label = row_2[0]
            else:
                continue

            if row_1[1] == row_2[1] or row_1[1] == row_3[1]:
                image_label = row_1[1]
            elif row_2[1] == row_3[1]:
                image_label = row_2[1]
            else:
                continue

            if text_label == 'positive' and image_label == 'negative':
                continue
            if text_label == 'negative' and image_label == 'positive':
                continue

            if text_label == image_label:
                multi_label = text_label
            elif text_label == 'neutral':
                multi_label = image_label
            elif image_label == 'neutral':
                multi_label = text_label
            else:
                continue
            labels[index] = [text_label, image_label, multi_label]
            i += 1
        print(i)

        data_names = os.listdir(data_dir)
        data_names.sort()
        text_names = list(filter(lambda x: x.endswith('.txt'), data_names))
        img_names = list(filter(lambda x: x.endswith('.jpg'), data_names))
        nums = 0
        negative, neutral, positive = 0, 0, 0
        for i in range(len(img_names)):
            img_name = img_names[i]
            index = img_name.split(".")[0]
            if index in labels.keys():
                nums += 1
                pass
            else:
                continue
            path_img = os.path.join(data_dir, img_name)
            path_text = os.path.join(data_dir, text_names[i])
            data_info["texts_info"].append((path_text, label_name.index(labels[index][0])))
            data_info["images_info"].append((path_img, label_name.index(labels[index][1])))
            data_info["labels"].append(label_name.index(labels[index][2]))
            if label_name.index(labels[index][2]) == 0:
                neutral += 1
            elif label_name.index(labels[index][2]) == 1:
                negative += 1
            else:
                positive += 1
        print(negative, neutral, positive, nums)

        n = len(data_info["texts_info"])
        index = [i for i in range(len(data_info["texts_info"]))]
        np.random.shuffle(index)
        data_info["texts_info"] = np.array(data_info["texts_info"])[index]
        data_info["images_info"] = np.array(data_info["images_info"])[index]
        data_info["labels"] = np.array(data_info["labels"])[index]
        if mood == "valid":
            data_info["texts_info"] = data_info["texts_info"][int(n * 0.9):]
            data_info["images_info"] = data_info["images_info"][int(n * 0.9):]
            data_info["labels"] = data_info["labels"][int(n * 0.9):]
        elif mood == "train":
            data_info["texts_info"] = data_info["texts_info"][:int(n * 0.7)]
            data_info["images_info"] = data_info["images_info"][:int(n * 0.7)]
            data_info["labels"] = data_info["labels"][:int(n * 0.7)]
        else:
            data_info["texts_info"] = data_info["texts_info"][int(n * 0.7):int(n * 0.9)]
            data_info["images_info"] = data_info["images_info"][int(n * 0.7):int(n * 0.9)]
            data_info["labels"] = data_info["labels"][int(n * 0.7):int(n * 0.9)]
        return data_info


def collate_fn(batch):
    """
    此函数的作用是定义如何取样本
    """
    MAX_LEN = 128

    lens = [min(len(row["text_id"]), MAX_LEN) for row in batch]
    batch_size, max_seq_len = len(batch), max(lens)
    text_tensor = torch.zeros((batch_size, max_seq_len))
    mask_tensor = torch.zeros((batch_size, max_seq_len))
    image_tensor = torch.zeros((batch_size, 3, 224, 224))
    label_tensor = torch.zeros((batch_size, 1))
    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch] = input_row["text_id"]
        mask_tensor[i_batch] = input_row["text_mask"]
        image_tensor[i_batch] = input_row["img"]
        label_tensor[i_batch] = input_row["label"]

    return text_tensor, mask_tensor, image_tensor, label_tensor



if __name__ == "__main__":
    np.random.seed(111)
    dataset = MVSADataset("../data/MVSA/data", "../data/MVSA/labelResultAll.txt", dataset='multi',
                          transform=process, mood='valid')
    data = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    n = len(data)
    for text_id, text_mask, image, label in data:
        print(text_id.size(), text_mask.size(), image.size(), label.size())



