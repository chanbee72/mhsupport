import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, logging
import spacy
import math
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from captum.attr import IntegratedGradients
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.set_verbosity_error()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base').eval()
roberta.to(device)
for name, param in roberta.named_parameters():
    param.requires_grad = False

nlp = spacy.load('en_core_web_sm')

basedata = pd.read_csv('basedata.csv')
train_data, test_data = train_test_split(basedata, test_size=0.2, random_state=99)



class model(nn.Module):
    def __init__(self, in_size, hid_size, out_size, sentence_num, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_size*sentence_num, hid_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hid_size, hid_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hid_size, out_size)

        self.sentence_num = sentence_num

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3).to(device)

        self.pos_encoder = PositionalEncoding(d_model=768, dropout=dropout, max_len=100)

    def forward(self, inputs):
        emb_vec = self.get_embedding(inputs)

        h = self.relu1(self.linear1(emb_vec))
        h = self.dropout1(h)
        h = self.relu2(self.linear2(h))
        h = self.dropout2(h)
        h = self.linear3(h)

        return h

    def get_embedding(self, comments):
        vectors = torch.empty((0,self.sentence_num*768)).to(device) # num_comment * (max_sentence_num*768)
        for comment in comments:
            sents = self.text2sent(comment)
            embs = self.sent2emb(sents) # num_sentence * 768
            embs, masks = self.embedding_padding(embs) # embs: num_sentence * 768 masks: 1 * 85
            embs.to(device)
            masks.to(device)
            self.encoder.to(device)

            embs = embs.type(torch.float32)
            masks = masks.type(torch.float32)
            #embs = embs.unsqueeze(0) # 1 * max_sentence_num * 768
            embs = embs.unsqueeze(1) # max_sentence_num * 1 * 768
            encoding = self.encoder(embs, src_key_padding_mask=masks) # max_sentence_num * 1 * 768

            vector = torch.flatten(encoding)
            vector = vector.unsqueeze(0)

            vectors = torch.cat((vectors, vector))
        return vectors
            
    def text2sent(self, text):
        sentence_list = []

        doc = nlp(text)
        for sent in doc.sents:
            sentence_list.append(sent.text)

        return sentence_list

    def sent2emb(self, sents):
        encodings = tokenizer(sents, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        encodings.to(device)
        
        outputs = roberta(**encodings)
        embs = outputs[1] # num_sentence * 768 ( CLS token embedding )
        embs = self.pos_encoder(embs) # num_sentence * 768

        return embs

    def embedding_padding(self, embs):
        num_sentence = embs.size(0)
        if num_sentence > self.sentence_num:
            embs = embs[:self.sentence_num]
            num_sentence = self.sentence_num

            masks_ = torch.ones((self.sentence_num))
            masks_ = masks_.unsqueeze(0).to(device)

            return embs, masks_
        num_padding = self.sentence_num - num_sentence
        padding_emb = torch.zeros((num_padding, 768)).to(device)
        embs = torch.cat((embs, padding_emb)) # max_sentence_num * 768

        # masks = torch.ones((self.sentence_num, self.sentence_num))
        # masks[:, num_sentence:] = 0
        # masks[num_sentence:, :] = 0

        masks_ = torch.ones((num_sentence))
        masks_ = torch.cat((masks_, torch.zeros((num_padding))))
        masks_ = masks_.unsqueeze(0).to(device)

        return embs, masks_



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # max_len * 1
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model) # max_len * 768
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self. register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



def get_max_sent_num(comments):
    max_sent_num = 0
    for comment in comments:
        doc = nlp(comment)
        sent_num = len(list(doc.sents))
        if max_sent_num < sent_num: max_sent_num = sent_num

    return max_sent_num


def label_onehot(labels):
    idx_dict = { x:i for i, x in enumerate(set(labels)) }
    onehot = []

    for label in labels:
        vec = [0]*len(idx_dict)
        vec[idx_dict[label]] = 1
        onehot.append(vec)

    return onehot


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        item = {'texts': self.texts[idx]}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def train(dataloader, model, optimizer, loss_fn):
    model.train()

    epoch_loss = 0
    size = len(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        texts = batch['texts']
        labels = batch['labels'].to(device)

        outputs = model(texts)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('Training loss: {:.3f}'.format(epoch_loss/size))


def test(dataloader, model, loss_fn, max_acc):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, acc = 0, 0


    with torch.no_grad():
        y_pred = torch.empty((0)).cpu()
        y_true = torch.empty((0)).cpu()

        for batch in dataloader:
            texts = batch['texts']
            labels = batch['labels'].to(device)

            outputs = model(texts)

            test_loss += loss_fn(outputs, labels).item()
            acc += (outputs.softmax(1).argmax(1) == labels).type(torch.float).sum().item()

            y_pred = torch.cat((y_pred, outputs.softmax(1).argmax(1).cpu()))
            y_true = torch.cat((y_true, labels.cpu()))

        test_loss /= size
        acc /= size

        #y_pred = outputs.softmax(1).argmax(1).cpu()
        # y_true = labels.cpu()

        print('Test loss: {:.3f}, Accuracy: {:.3f}%'.format(test_loss, acc*100))
        #print(precision_recall_fscore_support(y_true, y_pred, average=None, labels =[0, 1, 2]))
        if max_acc <= acc:
            torch.save(model.state_dict(), './btf_is_{}.pt'.format(model.sentence_num))
            max_acc = acc

    return y_true, y_pred, max_acc



# max_sent_num = get_max_sent_num(basedata['comment_text'].tolist())
sent_num = 10

model = model(768, 768, 3, sentence_num=sent_num, dropout=0.2)
model.to(device)

label_vec = label_onehot((train_data['is_score']).tolist())
label_vec = torch.tensor(label_vec, dtype=torch.float)

train_dataset = Dataset(train_data['comment_text'].tolist(), label_vec)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = Dataset(test_data['comment_text'].tolist(), (test_data['is_score']-1).tolist())
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 50
max_acc = 0

for i in tqdm(range(num_epochs)):
    print("Epoch {:}".format(i+1))
    train(train_dataloader, model, optimizer, loss_fn)
    y_true, y_pred, max_acc = test(test_dataloader, model, loss_fn, max_acc)


    result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2])

    print('is_score  :', [0,1,2])
    print('precision :', result[0])
    print('recalss   :', result[1])
    print('fscore    :', result[2])
    print('support   :', result[2])
    print('='*50)

print('max accuracy: {}'.format(max_acc))
