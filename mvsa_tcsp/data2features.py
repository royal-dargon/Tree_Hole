from transformers import BertTokenizer

bert_en_model = "../pre_model/pretrained_berts/bert_en"
tokenizer = BertTokenizer.from_pretrained(bert_en_model)
# r = tokenizer.tokenize("From Home Work to Modern Manufacture. Modern manufacturing has changed over time.")


# create the list of word list
class Vocab:
    def __init__(self, vocab_path):
        self.UNK = '[UNK]'
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(self.UNK))

    def __len__(self):
        return len(self.itos)


# test the word list
vocab = Vocab("../pre_model/pretrained_berts/bert_en/vocab.txt")
print(vocab.stoi['good'])


def text2features(text_rows):
    print(len(text_rows))
    for row in text_rows:
        inputs_id = tokenizer.encode(
            row,
            add_special_tokens=True,
            max_length=100,
            padding='longest',
            return_tensors='pt'
        )
        print(inputs_id)
    return 1
