import torch
import gc
gc.collect()
torch.cuda.empty_cache()

from torch import nn, Tensor
import pandas as pd
import numpy as np
from transformers import RobertaModel, RobertaTokenizer, logging
from sklearn.model_selection import train_test_split
import spacy
from torch.utils.data import DataLoader
import math
import pickle
from tqdm import tqdm
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.set_verbosity_error()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.to(device)
for name, param in roberta.named_parameters():
    param.reauires_grad = False

nlp = spacy.load('en_core_web_sm')

basedata = pd.read_csv('basedata.csv')
train_data, test_data = train_test_split(basedata, test_size=0.2, random_state=99)
train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=99)



class cont_model(nn.Module):
    def __init__(self, in_size, hid_size, out_size, train_model, dropout=0.2):
        super().__init__()
        self.train_model = train_model
        self.linear1 = nn.Linear(in_size*train_model.sentence_num*3, hid_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hid_size, hid_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hid_size, out_size)

    def forward(self, inputs1, inputs2):
        embedding1 = self.processing_emb(inputs1)    # num_comment * (max_sentence_num*768)
        embedding2 = self.processing_emb(inputs2)
        new_embedding = torch.cat((embedding1, embedding2), dim=1)
        new_embedding = torch.cat((new_embedding, torch.sub(embedding1, embedding2)), dim=1)

        h = self.relu1(self.linear1(new_embedding))
        h = self.dropout1(h)
        h = self.relu2(self.linear2(h))
        h = self.dropout2(h)
        h = self.linear3(h)

        return h

    def processing_emb(self, input_embs):
        vectors = torch.empty((0, self.train_model.sentence_num*768)).to(device)

        for input_emb in input_embs:
            emb = self.train_model.pos_encoder(input_emb)
            #embs, masks = self.train_model.embedding_padding(embs)
            mask = self.get_masking(input_emb)

            emb.to(device)
            mask.to(device)
            self.train_model.encoder.to(device)
        

            emb = emb.type(torch.float32)
            mask = mask.type(torch.float32)

            emb = emb.unsqueeze(1)
            encoding = self.train_model.encoder(emb, src_key_padding_mask=mask)

            v = torch.flatten(encoding)
            v = v.unsqueeze(0)

            vectors = torch.cat((vectors, v))

        return vectors

    def get_masking(self, embs):
        f = embs != 0
        f = f.sum(dim=1)
        f = f > 0
        num_sentence = f.sum()

        num_padding = self.train_model.sentence_num - num_sentence

        #masks = torch.ones((self.train_model.sentence_num, self.train_model.sentence_num))
        #masks[:, num_sentence:] = 0
        #masks[num_sentence:, :] = 0

        masks_ = torch.ones((num_sentence))
        masks_ = torch.cat((masks_, torch.zeros((num_padding))))
        masks_ = masks_.unsqueeze(0).to(device)

        return masks_



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
            embs, masks = self.embedding_padding(embs) # embs: num_sentence * 768 masks: 85 * 85
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

        #masks = torch.ones((self.max_sentence_num, self.max_sentence_num))
        #masks[:, num_sentence:] = 0
        #masks[num_sentence:, :] = 0

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
        
        if sent_num==85:
            print(comment)
        
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
    def __init__(self, texts1, texts2, labels):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels

    def __getitem__(self, idx):
        item = {'texts1': self.texts1[idx]}
        item['texts2'] = self.texts2[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def make_cmt_pair(comment_keys, labels, num=1):
    cmt_key_pairs = []
    pair_labels = []

    pair2label = {(0,0):0, (0,1):1, (0,2):2, (1,0):3, (1,1):4, (1,2):5, (2,0):6, (2,1):7, (2,2):8}

    for i, cmt_key1 in enumerate(comment_keys):
        for j, cmt_key2 in enumerate(comment_keys):
            if i==j: continue

            pair = (cmt_key1, cmt_key2)
            label = pair2label[(labels[i], labels[j])]
            cmt_key_pairs.append(pair)
            pair_labels.append(label)

    print('total comment pair:', len(cmt_key_pairs))

    # cmt_key_pairs : [(key1, key2), ...]
    # pair_labels : [label1, label2, ...]
    # pair2label : { (key1's is_score, key2's is_socre):label, ... }


    label2score = {0:0, 1:-1, 2:-2, 3:1, 4:0, 5:-1, 6:2, 7:1, 8:0}


    if num%6 != 0:

        pair_labels = [label2score[l]+2 for l in pair_labels]
        
        pairs_dict = { key_pair:label for (key_pair, label) in zip(cmt_key_pairs, pair_labels) }

        random.shuffle(cmt_key_pairs)
        pair_labels = [pairs_dict[key_pair] for key_pair in cmt_key_pairs]
        print('Return whole comment pair')
        return cmt_key_pairs, pair_labels

    
    pair_labels = np.array(pair_labels)

    new_cmt_key_pairs = []
    new_pair_labels = []

    print('Balancing...')
    print("(key1's is_score, key2's is_score) num_pairs --> new_score num_pairs")
    for (pair, label) in pair2label.items():
        idx = np.where(pair_labels==label)[0]
        key_pairs = [cmt_key_pairs[i] for i in idx]

        print('Before -', pair, len(key_pairs))

        score = label2score[label]

        if score == 0:
            _num = num//3
        elif score == 1 or score == -1:
            _num = num//2
        else:
            _num = num
        
        random_idx = np.random.choice(len(key_pairs), _num, replace=False)
        print('After  -', score, len(random_idx))

        new_cmt_key_pairs += [key_pairs[i] for i in random_idx]
        new_pair_labels += [score+2]*_num
    
    
    pairs_dict = { key_pair:label for (key_pair, label) in zip(new_cmt_key_pairs, new_pair_labels) }

    random.shuffle(new_cmt_key_pairs)
    new_pair_labels = [pairs_dict[key_pair] for key_pair in new_cmt_key_pairs]

    print("# of comment pairs : {}".format(len(new_pair_labels)))
    return new_cmt_key_pairs, new_pair_labels

def label2vec(labels):
    idx_dict = { x:i for i, x in enumerate(set(labels)) }
    onehot = []

    for label in labels:
        vec = [0]*len(idx_dict)
        vec[idx_dict[label]] = 1
        onehot.append(vec)

    onehot = torch.tensor(onehot, dtype=float, device=device)

    return onehot

def key2emb(keys_pair, emb_dict):
    embs_pair = []
    for (key1, key2) in keys_pair:
        emb1 = emb_dict[key1][0]
        emb2 = emb_dict[key2][0]

        embs_pair.append((emb1, emb2))

    return embs_pair

def emb_padding(sent_num, embs):
    for i, emb in enumerate(embs):
        num_sent = emb.size(0)
        if num_sent > sent_num:
            emb = emb[:sent_num]
            
        else:
            num_padding = sent_num - num_sent

            padding = torch.zeros((num_padding, emb.size(1))).to(device)
            emb = torch.cat((emb, padding))
        embs[i] = emb
    return embs


def train(dataloader, model, optimizer, loss_fn):
    model.train()

    epoch_loss = 0
    size = len(dataloader.dataset)
    acc = 0

    for i, batch in enumerate(dataloader):
        texts1 = batch['texts1']
        texts2 = batch['texts2']
        labels = batch['labels'].to(device)

        outputs = model(texts1, texts2)

        #print('pred', outputs.softmax(1).argmax(1))
        #print('corr', labels.softmax(1).argmax(1))

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        acc += (outputs.softmax(1).argmax(1) == labels.softmax(1).argmax(1)).type(torch.float).sum().item()
    acc /= size

    print('Training Loss: {:.3f}, Accuracy: {:.3f}'.format(epoch_loss/size, acc*100))

def test(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, acc = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            texts1 = batch['texts1']
            texts2 = batch['texts2']
            labels = batch['labels'].to(device)

            outputs = model(texts1, texts2)
            print(outputs.softmax(1).argmax(1))

            test_loss += loss_fn(outputs, labels).item()
            acc += (outputs.softmax(1).argmax(1) == labels).type(torch.float).sum().item()

        test_loss /= size
        acc /= size

        print('Test Loss: {:.3f}, Accuracy: {:.3f}%'.format(test_loss, acc*100))

#max_sent_num = get_max_sent_num(basedata['comment_text'].tolist())
sent_num = 20


btf = model(768, 768, 3, sentence_num=sent_num, dropout=0.2)
btf.to(device)
btf.load_state_dict(torch.load('btf_is_{}.pt'.format(sent_num)))

cont_model = cont_model(768, 768, 5, btf, dropout=0.2)
cont_model.to(device)

with open('cmt_roberta_emb.pickle', 'rb') as f:
    embs_dict = pickle.load(f)

#print(len(train_data)) # 252
#print(len(test_data))  # 64

print('Make train pairs...')
comment_key_pairs, labels = make_cmt_pair(train_data['comment_key'].tolist(), (train_data['is_score']-1).tolist(), 6*1000)
labels = label2vec(labels)
cmt_emb_pairs = key2emb(comment_key_pairs, embs_dict)

emb1 = [pair[0] for pair in cmt_emb_pairs]
emb2 = [pair[1] for pair in cmt_emb_pairs]
emb1 = emb_padding(sent_num, emb1)
emb2 = emb_padding(sent_num, emb2)
# emb1 = emb1[:20]
# emb2 = emb2[:20]

train_dataset = Dataset(emb1, emb2, labels)
train_dataloader = DataLoader(train_dataset, batch_size=8)

print('-'*50)
print('Make test pairs...')
comment_key_pairs, labels = make_cmt_pair(valid_data['comment_key'].tolist(), (valid_data['is_score']-1).tolist())
cmt_emb_paris = key2emb(comment_key_pairs, embs_dict)


emb1 = [pair[0] for pair in cmt_emb_pairs]
emb2 = [pair[1] for pair in cmt_emb_pairs]
emb1 = emb_padding(sent_num, emb1)
emb2 = emb_padding(sent_num, emb2)

test_dataset = Dataset(emb1, emb2, labels)
test_dataloader = DataLoader(test_dataset, batch_size=8)


optimizer = torch.optim.Adam(cont_model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 30
max_acc = 0

# traind_model(btf model) linear layer Freezing
for i, (name, param) in enumerate(cont_model.named_parameters()):
    if i <= 5:
        param.requires_grad = False
    

print('='*50)
for i in tqdm(range(num_epochs)):
    print('Epoch {:}'.format(i+1))
    train(train_dataloader, cont_model, optimizer, loss_fn)
    test(test_dataloader, cont_model, loss_fn)
