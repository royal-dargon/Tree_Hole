# this is the file about config and utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

config = {
    "text_model": {}
}

config["text_model"]["epoch"] = 1


def compute_acc(out, y):
    acc_nums = 0
    _, index = torch.topk(out, 1, dim=1, largest=True, sorted=True)
    _, index_real = torch.topk(y, 1, dim=1, largest=True, sorted=True)
    res_index = 0
    for res in index:
        if res == index_real[res_index]:
            acc_nums += 1
        res_index += 1
    return acc_nums
