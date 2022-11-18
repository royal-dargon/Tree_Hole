# 主要的作用是提前将文本数据和图片数据提取出来变成hdf5格式节约运行的时间和成本
import h5py

from transformers import BertTokenizer
import numpy as np

import pre_data

file = h5py.File("../data/HDF5_DATA/data_single.hdf5", "w")
text_group = file.create_group("text")
image_group = file.create_group("vision")
label_group = file.create_group("labels")

bert_en_model = "../pre_model/pretrained_berts/bert_en"
tokenizer = BertTokenizer.from_pretrained(bert_en_model)


def write2hdf5(data):
    """主要的结构是先是分为了三类数据的表"""
    texts, images, label_text, label_image = data["text"], data["image"], data["text_labels"], data["image_labels"]
    text_contents = pre_data.text2id(texts)
    text_group.create_dataset("text_id", data=text_contents[0].numpy())
    text_group.create_dataset("text_mask", data=text_contents[1].numpy())
    image_group.create_dataset("image", data=np.array(images))
    label_text = pre_data.label2features(label_text)
    label_image = pre_data.label2features(label_image)
    label_group.create_dataset("text_label", data=np.array(label_text))
    label_group.create_dataset("image_label", data=np.array(label_image))
    print(file["labels"]["text_label"][:])


def main():
    data = pre_data.get_single()
    write2hdf5(data)


if __name__ == "__main__":
    main()



