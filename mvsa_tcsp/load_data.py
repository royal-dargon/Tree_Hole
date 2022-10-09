import pandas as pd
import os
import cv2
import numpy as np


def load_mvsa_single():
    filepath = '../data/MVSA_Single/data'
    label_path = '../data/MVSA_Single/labelResultAll.txt'
    file_name = os.listdir(filepath)
    file_name.sort()                # sure the rank about files
    text_rows = []
    image_rows = []
    label_rows = []
    for file in file_name:
        if file.endswith("txt"):
            f = open(filepath + "/" + file, encoding='unicode_escape')
            lines = f.readlines()   # 确定一共是4869条数据
            text_rows.append(lines)
        elif file.endswith("jpg"):
            # three loads [length, width, 3]
            i = cv2.imread(filepath + "/" + file)
            image_rows.append(i)
    f = open(label_path)
    lines = f.readlines()
    i = 0
    text_label, image_label = [], []
    for line in lines:
        line = line.split(sep="\t")
        line = line[1].replace('\n', '')
        line = line.split(",")
        if i == 0:
            i += 1
            continue
        i += 1
        text_label.append(line[0])
        image_label.append(line[1])
    label_rows.append(text_label)
    label_rows.append(image_label)

    return text_rows, image_rows, label_rows


def load_data(data_name):
    if data_name == "all":
        pass
    elif data_name == "single":
        text_rows, image_rows, label_rows = load_mvsa_single()
        return text_rows, image_rows, label_rows
    else:
        print("the dataset is not exist")

