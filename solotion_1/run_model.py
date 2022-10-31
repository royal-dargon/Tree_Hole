
import torch
import torch.nn as nn
import torch.optim as optim


import model
import pre_data
from utils import cpu, device

if __name__ == "__main__":
    # 获取文件夹的源数据
    data = pre_data.get_single()

    # 划分数据
    data_len = len(data["text"])
    res_data = pre_data.divide_data(data, batch_size=64, data_length=data_len)

    # 确定是否使用gpu和一些其他的超参数
    is_use_gpu = False
    epoch = 1
    print(device)
    TextModel = model.Text2Features(device, use_gpu=is_use_gpu)
    if is_use_gpu:
        TextModel = TextModel.to(device)
    text_optim = optim.Adam([param for param in TextModel.parameters()], lr=1e-5)
    loss_func_text = nn.CrossEntropyLoss()
    # 文本模型的训练
    for i in range(epoch):
        # 注意一下标签的序号
        j = 0
        losses = 0
        plot_losses = []
        for row in res_data["train"]["text"][:5]:
            text_optim.zero_grad()
            text = pre_data.text2id(row)
            out = TextModel(text)
            out = out.to(cpu)
            text_label = res_data["train"]["text_label"][j]
            j += 1
            y = pre_data.label2features(text_label)
            y = y.squeeze(1)
            loss = loss_func_text(out, y)
            loss.backward()
            text_optim.step()
            losses += loss
        plot_losses.append(losses / j)
        print("the %d epoch losses is %f" % (i + 1, losses / j))

    # 保存模型的参数
    torch.save(TextModel.state_dict(), './save/single/textmodel.pt')

