import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, logging
import spacy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from captum.attr import IntegratedGradients


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
    def __init__(self, in_size, hid_size, out_size, max_sentence_num, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_size*max_sentence_num, hid_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hid_size, hid_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hid_size, out_size)

        self.max_sentence_num = max_sentence_num

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.pos_encoding = self.get_sinusoid_encoding_table(self.max_sentence_num*512, 768)
        self.pos_encoding = torch.tensor(self.pos_encoding)
        self.nn_pos = nn.Embedding.from_pretrained(self.pos_encoding, freeze=True)

    def forward(self, inputs):
        emb_vec = self.get_embedding(inputs)

        h = self.relu1(self.linear1(emb_vec))
        h = self.dropout1(h)
        h = self.relu2(self.linear2(h))
        h = self.dropout2(h)
        h = self.linear3(h)

        return h

    def get_embedding(self, comments):
        vectors = torch.empty((0,self.max_sentence_num*768)).to(device) # num_comment * (max_sentence_num*768)
        for comment in comments:
            sents = self.text2sent(comment)
            embs = self.sent2emb(sents) # num_sentence * 512 * 768
            embs, masks = self.embedding_padding(embs)
          
            embs = embs.type(torch.float32)
            print(masks.size())
            encoding = self.encoder(embs, masks) # max_sentence_num * 512 * 768
            cls_encoding = encoding[:,0,:]

            vector = torch.flatten(cls_encoding)
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
        input_embs = outputs[0] # num_sentence * 512(tokenizer max_length) * 768
        #input_embs = outputs[1] # num_sentence * 768 ( CLS token embedding )

        inputs = encodings['input_ids'] # num_sentence * 512

        positions = torch.empty((0, 512)).to(device)
        s = 1
        for i in range(len(sents)):
            pos = torch.arange(s, s+512).to(device)
            pos_mask = inputs[i].eq(1)
            pos.masked_fill_(pos_mask, 0)
            pos = pos.unsqueeze(0)
            positions = torch.cat((positions, pos))
            s = pos.max()
        positions = positions.to(torch.int64)
        
        self.nn_pos.to(device)
        positions.to(device)
        pos_embs = self.nn_pos(positions) # num_sentence * 512 * 768

        embeddings = input_embs + pos_embs # num_sentence * 512 * 768

        return embeddings

    def embedding_padding(self, embs):
        num_sentence = embs.size()[0]
        num_padding = self.max_sentence_num - num_sentence
        padding_emb = torch.zeros((num_padding,512,768)).to(device)
        embs = torch.cat((embs, padding_emb))
        
        masks = torch.logical_not(embs.eq(0))

        return embs, masks


    def get_sinusoid_encoding_table(self, n_seq, d_hidn):
        def cal_angle(position, i_hidn):
            return position / np.power(10000, 2*(i_hidn//2) / d_hidn)
        def get_posi_angle_vec(position):
            return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]
        
        sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return sinusoid_table


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
        for batch in dataloader:
            texts = batch['texts']
            labels = batch['labels'].to(device)

            outputs = model(texts)

            test_loss += loss_fn(outputs, labels).item()
            acc += (outputs.softmax(1).argmax(1) == labels).type(torch.float).sum().item()

        test_loss /= size
        acc /= size

        y_true = outputs.softmax(1).argmax(1).cpu()
        y_pred = labels.cpu()

        print('Test loss: {:.3f}, Accuracy: {:.3f}%'.format(test_loss, acc*100))
        #print(precision_recall_fscore_support(y_true, y_pred, average=None, labels =[0, 1, 2]))
        if max_acc <= acc:
            torch.save(model.state_dict(), './model.pt')
            max_acc = acc

    return y_true, y_pred, max_acc



max_sent_num = get_max_sent_num(basedata['comment_text'].tolist())

model = model(768, 768, 3, max_sent_num, dropout=0.2)
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

for i in range(num_epochs):
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

    break


