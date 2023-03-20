import os

from torch.utils.data import Dataset


label_name = ['neutral', 'negative', 'positive']


class MVSADataset(Dataset):
    def __init__(self, data_dir, label_path, dataset='single'):
        self.data_dir = data_dir
        self.label_path = label_path
        self.dataset = dataset
        if dataset == 'single':
            pass
        else:
            self.data_info = self.get_multi_data_info(data_dir, label_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        path_text, path_img, text_label, img_label, multi_label = self.data_info["texts_info"][index][0], \
                                                                  self.data_info["images_info"][index][0], \
                                                                  self.data_info["texts_info"][index][1], \
                                                                  self.data_info["images_info"][index][1], \
                                                                  self.data_info["labels"][index]
        print(path_text, path_img, text_label, img_label, multi_label)
        data, label = (path_text, path_img), (text_label, img_label, multi_label)

        return data, label

    @staticmethod
    def get_multi_data_info(data_dir, label_path):
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
        return data_info


a = MVSADataset("../data/MVSA/data", "../data/MVSA/labelResultAll.txt", dataset='multi')
m, n = a.__getitem__(100)




