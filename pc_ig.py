import torch
from torch import nn, Tensor
from transformers import RobertaModel, RobertaTokenizer, logging, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import spacy
import pc_model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from captum.attr import LayerIntegratedGradients, IntegratedGradients
from tqdm import tqdm
import math
import torch.nn.functional as F


torch.set_printoptions(profile='full')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

basedata = pd.read_csv('basedata.csv')
_, test_data = train_test_split(basedata, test_size=0.2, random_state=99)

nlp = spacy.load('en_core_web_sm')



class Dataset(torch.utils.data.Dataset):
    def __init__(self, posts, comments, labels):
        self.posts = posts
        self.comments = comments
        self.labels = self.label2vec(labels)

    def __getitem__(self, idx):
        item = {'post': self.posts[idx]}
        item['comment'] = self.comments[idx]
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def label2vec(self, labels):
        idx_dict = {x:i for i, x in enumerate(set(labels))}
        onehot = []
    
        for label in labels:
            vec = [0]*len(idx_dict)
            vec[idx_dict[label]] = 1
            onehot.append(vec)

        onehot = torch.tensor(onehot, dtype=torch.float, device=device)

        return onehot


def ig_forward(inputs):
    # inputs : post sentence_embedding + comment sentence_embedding
    
    h = model.relu(model.linear1(inputs))
    h = model.relu(model.linear2(h))
    h = model.linear3(h)

    return h


def ig_forward_sm(inputs):

    h = model.relu(model.linear1(inputs))
    h = model.relu(model.linear2(h))
    h = F.softmax(model.linear3(h), dim=-1)

    return h


"""
def get_sbert_encoding(post, comment):
    for i, text in enumerate((post, comment)):
        sent_num = 20 if i==0 else 10
        
        sents = []
        doc = nlp(text[0])
        for sent in doc.sents:
            sents.append(sent.text)
            if len(sents) >= sent_num: break

        
        if i == 0:
            post_sbert_encoding = model.sbert.encode(sents, convert_to_tensor=True)
        else:
            comment_sbert_encoding = model.sbert.encode(sents, convert_to_tensor=True)

    return post_sbert_encoding, comment_sbert_encoding


def ig_forward2(inputs):
    # inputs : (post sbert encoding, comment sbert encoding)
    post_sbert_encoding, comment_sbert_encoding = inputs
    
    post_pos_embedding = model.pos_encoder(post_sbert_encoding)
    commnet_pos_embedding = model.pos_encoder(comment_sbert_encoding)


    for i, emb in enumerate([post_pos_embedding, commnet_pos_embedding]):
        sent_num = 20 if i==0 else 10
        
        num_padding = sent_num - emb.shape[0]
        padding = torch.zeros((num_padding, 384)).to(device)
        embedding = torch.cat((emb, padding))
        
        mask = torch.ones(sent_num).to(device)
        mask[-num_padding:]=0
        mask = mask.unsqueeze(0)

        embedding = embedding.unsqueeze(1)
        encoding = model.sent_encoder(embedding, src_key_padding_mask=mask)

        embedding = encoding.flatten()
        embedding = embedding.unsqueeze(0)

        if i == 0:
            post_embedding = embedding
        else:
            comment_embedding = embedding

    emb = torch.cat((post_embedding, comment_embedding), dim=1)
    h = ig_forward(emb)

    return h
"""

def interprete_text(model, post, comment, label):
    model.zero_grad()

    pred = model(post, comment).softmax(1).argmax(1)
    print("Predict : {}".format(pred.item()))
    print("Label   : {}".format(label.argmax(1).item()))
    print()
    embs = model.get_embedding(post, comment) 

    attribution = ig.attribute(embs, target=label.argmax(1))
    
    embs = embs.view(30, -1)
    attr = attribution.view(30, -1)

    post_sents = text2sent(post[0], p_sent_num)
    comment_sents = text2sent(comment[0], c_sent_num)

    for i in range(30):
        if i<20:
            print('POST Sentence {}'.format(i+1))
            print('(', post_sents[i], ')')
        else:
            print('COMMENT Sentence {}'.format(i+1-20))
            print('(', comment_sents[i-20], ')')

        print('>>> Input Vector : {}'.format(sum(embs[i])))
        print('>>> IG Vector    : {}'.format(sum(attr[i])))
    print()

    return attribution


def text2sent(text, sent_num):
    sents = []
    doc = nlp(text)
    for sent in doc.sents:
        sents.append(sent.text)

    l = len(sents)
    if l < sent_num:
        for i in range(sent_num-l):
            sents.append('PADDING {}'.format(i+1))

    return sents



def ig_process(dataloader, ig, file_name, norm=False):
    
    post_texts = []
    comment_texts = []
    best_sents = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            #if i==1: break

            print('{} EXAMPLE'.format(i+1))
            post = batch['post']
            comment = batch['comment']
            label = batch['label'].to(device)

            print('----------POST TEXT----------')
            print(post[0])
            print('-------COMMENT TEXT----------')
            print(comment[0])
            print()

            att = interprete_text(model, post, comment, label)
            att = att.view([1, -1, att.size(1)//(p_sent_num+c_sent_num)])
            
            if norm == True:
                att = F.normalize(att)
            
            att = torch.abs(att)
            snt_score = att.sum(dim=-1).squeeze()
            idx = torch.argmax(snt_score)
        
            post_sents = text2sent(post[0], p_sent_num)
            comment_sents = text2sent(comment[0], c_sent_num)

            if idx < 20:
                more = 'POST'
                sent = post_sents[idx]
            else:
                more = 'COMMENT'
                idx -= 20
                sent = comment_sents[idx]
        
            post_texts.append(post[0])
            comment_texts.append(comment[0])
            best_sents.append(sent)
        
            print('More Impact - {} {}'.format(more, idx+1))
            print('Sentence    -', sent)
            print('='*80)
    

    df = pd.DataFrame()
    df['post'] = post_texts
    df['comment'] = comment_texts
    df['best_sentence'] = best_sents

    df.to_csv('./ig_{}.csv'.format(file_name), index=False)



score_type = 'is_score'

post_text = test_data['post_text'].tolist()
cmt_text = test_data['comment_text'].tolist()
score = (test_data[score_type]-1).tolist()

dataset = Dataset(post_text, cmt_text, score)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

p_sent_num = 20
c_sent_num = 10
t_type = ('snt', 'snt', p_sent_num, c_sent_num)
in_size = p_sent_num*384 + c_sent_num*384
hid_size = 768
out_size = 3

model = pc_model.model(in_size, hid_size, out_size, t_type=t_type, dropout=0.2)
model.load_state_dict(torch.load('./pc_model.pt'))
model.to(device)
model.eval()


ig = IntegratedGradients(ig_forward)
ig_sm = IntegratedGradients(ig_forward_sm)

basic_ver = ig_process(dataloader, ig, file_name='basic_ver', norm=False)
softmax_ver = ig_process(dataloader, ig_sm, file_name='softmax_ver', norm=False)
normalization_ver = ig_process(dataloader, ig, file_name='normalization_ver', norm=True)
