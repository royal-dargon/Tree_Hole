# this is the file about config and utils
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

config = {
    "text_model": {},
    "image_model": {},
    "multi_model": {}
}

config["text_model"]["epoch"] = 1
config["text_model"]["train"] = False

config["image_model"]["epoch"] = 1
config["image_model"]["train"] = False

config["multi_model"]["epoch"] = 1
config["multi_model"]["train"] = True


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


# 对图像进行裁剪
def cut_image(image_rows):
    process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    images = []
    for row in image_rows:
        res = process(row)
        images.append(res.tolist())
    return torch.tensor(images)

