import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

config = {
    "text_model": {}
}

config["text_model"]["epoch"] = 100
