import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


import model
import pre_data
from utils import cpu, device, config, compute_acc


def train_textmodel(train_loader, val_loader, text_model, optimizer, loss_func):
    """train the text model"""
    tr_losses = []
    val_losses = []
    val_acc = []
    for epoch in tqdm(range(config["text_model"]["epoch"]), desc="training the text model ..."):
        start = time.time()
        train_index, val_index = 0, 0
        train_size, val_size = 0, 0
        losses = 0
        for row in tqdm(train_loader["text"]):
            batch_size = len(row)
            train_size += batch_size
            optimizer.zero_grad()
            text = pre_data.text2id(row)
            out = text_model(text)
            out = out.to(cpu)
            # print(out.shape)
            text_label = train_loader["text_label"][train_index]
            train_index += 1
            y = pre_data.label2features(text_label)
            y = y.squeeze(1)
            # print(y.shape)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            losses += loss.item() * batch_size
        tr_losses.append(losses / train_size)
        end = time.time()
        print("the %d epoch train losses is %f the time is %f s" % (epoch + 1, losses / train_size, (end - start)))

        losses = 0
        with torch.no_grad():
            for row in tqdm(val_loader["text"]):
                batch_size = len(row)
                val_size += batch_size
                text = pre_data.text2id(row)
                out = text_model(text)
                out = out.to(cpu)
                text_label = val_loader["text_label"][val_index]
                val_index += 1
                y = pre_data.label2features(text_label)
                y = y.squeeze(1)
                losses = loss_func(out, y)
                acc_nums = compute_acc(out, y)
                losses += loss.item() * batch_size
            val_losses.append(losses / val_size)
            val_acc.append(acc_nums/val_size)
            print("the %d epoch val losses is %f the acc is %f" % (epoch + 1, losses/val_size, acc_nums/val_size))

    # 保存模型的参数
    torch.save(TextModel.state_dict(), './save/single/textmodel.pt')


def test_textmodel(test_loader, text_model):
    """test the text model"""
    pass


if __name__ == "__main__":
    # 获取文件夹的源数据
    data = pre_data.get_single()

    # 划分数据
    data_len = len(data["text"])
    res_data = pre_data.divide_data(data, batch_size=64, data_length=data_len)

    # 确定是否使用gpu和一些其他的超参数
    is_use_gpu = True
    print(device)
    TextModel = model.Text2Features(device, use_gpu=is_use_gpu)
    if is_use_gpu:
        TextModel = TextModel.to(device)
    text_optim = optim.Adam([param for param in TextModel.parameters()], lr=1e-3)
    loss_func_text = nn.CrossEntropyLoss()
    # scheduler_text = ReduceLROnPlateau(text_optim, mode='min', patience=5, factor=0.1, verbose=True)
    train_textmodel(train_loader=res_data["train"], val_loader=res_data["val"],
                    text_model=TextModel, optimizer=text_optim, loss_func=loss_func_text)



