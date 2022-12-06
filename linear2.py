import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, logging
import spacy


# sentence's score predict model
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
        
        h = self.relu2(self.linear2(h))
        return h



# linear1 element wise avg. model
class linear2_model(nn.Module):
    def __init__(self, s_model, in_size, hid_size, out_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hid_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hid_size, hid_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hid_size, out_size)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nlp = spacy.load('en_core_web_sm')

        # s_model, roberta freezing
        self.s_model = s_model.eval()
        self.s_model.to(self.device)
        for name, param in self.s_model.named_parameters():
            param.requires_grad = False
        
        logging.set_verbosity_error()
        self.roberta = RobertaModel.from_pretrained('roberta-base').eval()
        self.roberta.to(self.device)
        for name, param in self.roberta.named_parameters():
            param.requires_grad = False
        
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    def forward(self, inputs):
        l2 = self.get_l2(inputs)
        h = self.relu1(self.linear1(l2))
        h = self.dropout1(h)
        h = self.relu2(self.linear2(h))
        h = self.dropout2(h)
        h = self.linear3(h)

        return h

    # get linear2's output in s_model
    def get_l2(self, texts):
        l2_outputs = torch.empty((0,512)).to(device)
        for text in texts:
            sents = self.text2sentence(text)
            emb = self.to_embeddings(sents)
            emb.to(self.device)

            s_output = self.s_model.linear_n(2, emb)
            s_output = torch.mean(s_output, axis=0)
            s_output = s_output.unsqueeze(0)

            l2_outputs = torch.cat((l2_outputs, s_output))

        return l2_outputs

    # text to sentence
    def text2sentence(self, text):
        sentence_list = []

        doc = self.nlp(text)
        for sent in doc.sents:
            sentence_list.append(sent.text)

        return sentence_list


    # sentence to embedding
    def to_embeddings(self, sentence_list):
        embeddings = torch.empty((0, 768)).to(self.device)
        for sentence in sentence_list:
            encoding = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            encoding.to(self.device)
            output = self.roberta(**encoding)
            embedding = output[1]
            embeddings = torch.cat((embeddings, embedding))

        return embeddings
            


# Dataset
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



# train & test
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



def test(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, acc = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            texts = batch['texts']
            labels = batch['labels'].to(device)

            outputs = model(texts)

            test_loss += loss_fn(outputs, labels).item()
            acc += (outputs.softmax(1).argmax(1) == labels).type(torch.float).sum().item()

        test_loss /= size
        acc /= size

        print('Test loss: {:.3f}, Accuracy: {:.3f}%'.format(test_loss, acc*100))



# setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

s_model = s_model(768, 512, 3)
# s_model.load_state_dict(torch.load('snt_model(is).pt'))
s_model.load_state_dict(torch.load('snt_model(es).pt'))

model = linear2_model(s_model=s_model, in_size=512, hid_size=256, out_size=3)
model.to(device)

# basedata = pd.read_csv('basedata_b.csv')
basedata = pd.read_csv('basedata.csv')
# train_data, test_data = train_test_split(basedata, random_state=99, test_size=0.2, stratify=basedata['is_score'])
train_data, test_data = train_test_split(basedata, random_state=99, test_size=0.2, stratify=basedata['es_score'])

# train_dataset = Dataset(train_data['comment_text'].tolist(), (train_data['is_score']-1).tolist())
train_dataset = Dataset(train_data['comment_text'].tolist(), (train_data['es_score']-1).tolist())
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_dataset = Dataset(test_data['comment_text'].tolist(), test_data['is_score']-1).tolisT())
test_dataset = Dataset(test_data['comment_text'].tolist(), (test_data['es_score']-1).tolist())
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 50



# training
for i in range(num_epochs):
    print("Epoch {:}".format(i+1))
    train(train_dataloader, model, optimizer, loss_fn)
    test(test_dataloader, model, loss_fn)
