import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, logging
import spacy



# setting
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

logging.set_verbosity_error()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.to(device)

nlp = spacy.load("en_core_web_sm")

basedata = pd.read_csv('basedata.csv')
train_data, test_data = train_test_split(basedata, test_size=0.2, random_state=99)



# model
class s_model(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hid_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hid_size, hid_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hid_size, out_size)

    def forward(self, inputs):        
        h = self.relu1(self.linear1(inputs))
        h = self.dropout1(h)
        h = self.relu2(self.linear2(h))
        h = self.dropout2(h)
        h = self.linear3(h)

        return h
    
    def linear_n(self, n, inputs):
        h = self.relu1(self.linear1(inputs))
        if n == 1:
            return h
        h = self.dropout1(h)
        h = self.relu2(self.linear2(h))
        return h



# dataloader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        item = {'embeddings': self.embeddings[idx]}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class snt_dataset(torch.utils.data.Dataset):
    def __init__(self, sent):
        self.sent = sent

    def __getitem__(self, idx):
        return self.sent[idx]

    def __len__(self):
        return len(self.sent)



# text to sentence
def text2sentence(text_data):
    sent_data = pd.DataFrame(columns=['comment_sent', 'is_score', 'es_score'])
    for i in range(len(text_data)):
        text = text_data.iloc[i, 0]
        is_ = text_data.iloc[i, 1]
        es_ = text_data.iloc[i, 2]

        doc = nlp(text)
        for sent in doc.sents:
            n = len(sent_data)
            sent_data.loc[n] = {'comment_sent': sent.text, 'is_score': is_, 'es_score': es_}

    return sent_data



train_data = train_data[['comment_text', 'is_score', 'es_score']]
train_sent = text2sentence(train_data)

test_data = test_data[['comment_text', 'is_score', 'es_score']]
test_sent = text2sentence(test_data)

train_sent_dataset = snt_dataset(train_sent['comment_sent'].tolist())
train_sent_dataloader = DataLoader(train_sent_dataset, batch_size=32)
test_sent_dataset = snt_dataset(test_sent['comment_sent'].tolist())
test_sent_dataloader = DataLoader(test_sent_dataset, batch_size=32)


# embedding
def to_embeddings(dataloader):
    embeddings = torch.empty((0, 768)).to(device)
    with torch.no_grad():
        for batch in dataloader:
            encoding = tokenizer(batch, return_tensors='pt', truancation=True, padding='max_length', max_length=512)
            encoding.to(device)
            output = roberta(**encoding)
            embedding = output[1]
            embeddings = torch.cat((embeddings, embedding))

    return embeddings

train_embeddings = to_embeddings(train_sent_dataloader)
test_embeddings = to_embeddings(test_sent_dataloader)



# train & test
def train(dataloader, model, optimizer, loss_fn):
    model.train()

    epoch_loss = 0
    size = len(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(embeddings)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('Traning loss: {:.3f}'.format(epoch_loss/size))


def test(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, acc = 0, 0

    with torch.no_grad():
        for batch in dataloader:         
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(embeddings)

            test_loss += loss_fn(outputs, labels).item()
            acc += (outputs.softmax(1).argmax(1) == labels).type(torch.float).sum().item()

        test_loss /= size
        acc /= size

        print('Test loss: {:.3f}, Accuracy: {:.3f}%'.format(test_loss, acc*100))

    return acc*100


# maim
model = s_model(in_size=768, hid_size=512, out_size=3)
model.to(device)

train_dataset = Dataset(train_embeddings, (train_sent['is_score']-1).tolist())
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = Dataset(test_embeddings, (test_sent['is_score']-1).tolist())
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 150
best_acc = 0

for i in range(num_epochs):
    print("Epoch: #{}".format(i+1))
    train(train_dataloader, model, optimizer, loss_fn)
    acc = test(test_dataloader, model, loss_fn)

    if best_acc <= acc:
        best_acc = acc
        torch.save(model.state_dict(), 'snt_model(is).pt')
