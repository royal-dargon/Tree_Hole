import load_data
import data2features
import model
import torch
import torch.nn as nn
import torch.optim as optim

# 图像特征是（width， height， 3）
text_rows, image_rows, label_rows = load_data.load_data("single")
print("data load successfully!")
text_features = data2features.text2features(text_rows[:20])
print("word to features successful")
image_features = data2features.image2features(image_rows[:20])
print("image to features successful")
text_label_features = data2features.label2features(label_rows[0][:20])

text_config = {
    'source_size': 768,
    'lstm_hidden_size': 2048,
    'lstm_num_layers': 1,
    'lstm_dropout': 0
}

batch_size = 10
text_model = model.TextModel(text_config, batch_size)
text_criterion = nn.CrossEntropyLoss()
text_optimizer = optim.Adam(text_model.parameters(), lr=1e-5)


n_items = 10
# 开始对文本情感分析的模型进行训练
for epoch in range(n_items):
    for i in range(0, text_features.shape[0], batch_size):
        if i + batch_size > text_features.shape[0]:
            # x = text_features[i:]
            # x = torch.tensor(x)
            continue
        else:
            x = text_features[i:i + batch_size]
            # x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            x = torch.tensor(x)
            y = text_label_features[i:i + batch_size]
            y = torch.tensor(y)
        output = text_model(x)
        loss = text_criterion(output, y)
        print(loss)
        loss.backward()
        text_optimizer.step()




