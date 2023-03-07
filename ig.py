import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from captum.attr import LayerIntegratedGradients, IntegratedGradients
from transformers import RobertaTokenizer, RobertaModel, logging
import spacy
import numpy as np
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.set_verbosity_error()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base').eval()
roberta.to(device)
for name, param in roberta.named_parameters():
    param.requires_grad = False

nlp = spacy.load('en_core_web_sm')

data = pd.read_csv('basedata.csv')


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
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6).to(device)

        self.pos_encoder = PositionalEncoding(d_model=768, dropout=dropout, max_len=max_sentence_num)

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
        num_padding = self.max_sentence_num - num_sentence
        padding_emb = torch.zeros((num_padding, 768)).to(device)
        embs = torch.cat((embs, padding_emb)) # max_sentence_num * 768

        masks = torch.ones((self.max_sentence_num, self.max_sentence_num))
        masks[:, num_sentence:] = 0
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

def custom_forward(inputs):
    h = model.relu1(model.linear1(inputs))
    h = model.relu2(model.linear2(h))
    h = model.linear3(h)

    return h



max_sent_num = get_max_sent_num(data['comment_text'].tolist())

model = model(768, 768, 3, max_sent_num, dropout=0.2).to(device)
model.load_state_dict(torch.load('btf_is.pt'))
model.eval()

ig = IntegratedGradients(custom_forward)

inputs = data['comment_text'].tolist()[:20]
labels = (data['is_score']-1).tolist()[:20]

dataset = Dataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=1)


def interprete_comment(model, comment, label=0):
    model.zero_grad()
    
    pred = model(comment).softmax(1).argmax(1)
    print('Predict:', pred.item())
    print('Label:', label.item()) 
    embs = model.get_embedding(comment)

    attribution = ig.attribute(embs, target=label)

    return attribution

"""
for i in range(len(inputs)):
    cmt = inputs[i]
    label = labels[i]
    
    cmt.to(device)
    pred = model(cmt).softmax(1).argmax(1)

    print(cmt)
    print(pred)

    #att = interprete_comment(model, cmt, label=label)
    #print(att)
"""

model.eval()

with torch.no_grad():
    for batch in dataloader:
        comment = batch['texts']
        label = batch['labels'].to(device)

        att = interprete_comment(model, comment, label)
        
        att = att.view([1, -1, att.size(1)//max_sent_num])
        snt_score = att.sum(dim=-1).squeeze()
        idx = torch.argmax(snt_score)

        sents = model.text2sent(comment[0])
        best_sent = 'PADDING SENTENCE' if len(sents) <= idx else sents[idx]
        
        print('Score:', snt_score[idx].item())
        print('Comment:', best_sent)
        print('='*50)
