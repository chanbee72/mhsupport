import torch
from torch import nn, Tensor
from transformers import RobertaModel, RobertaTokenizer, logging, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import spacy
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.set_verbosity_error()

nlp = spacy.load("en_core_web_sm")

class model(nn.Module):
    def __init__(self, in_size, hid_size, out_size, t_type=('all', 'all', 0, 0), dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hid_size)
        self.linear2 = nn.Linear(hid_size, hid_size)
        self.linear3 = nn.Linear(hid_size, out_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #self.roberta = RobertaModel.from_pretrained('roberta-base').eval()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.roberta.to(device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.bert = BertModel.from_pretrained('bert-base-uncased').eval()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2').eval()
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.sbert.to(device)

        self.post_type = t_type[0]
        self.comment_type = t_type[1]
        self.post_num = t_type[2]
        self.comment_num = t_type[3]

        #self.freezing(self.roberta)
        #self.freezing(self.bert)
        #self.freezing(self.sbert)

        self.sent_encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=4)
        self.sent_encoder = nn.TransformerEncoder(self.sent_encoder_layer, num_layers=3).to(device)

        self.seg_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.seg_encoder = nn.TransformerEncoder(self.seg_encoder_layer, num_layers=3).to(device)


        self.pos_encoder = PositionalEncoding(d_model=384, dropout=dropout, max_len=100)


    def forward(self, posts, comments):
        emb = self.get_embedding(posts, comments)

        h = self.relu(self.linear1(emb))
        h = self.dropout(h)
        h = self.relu(self.linear2(h))
        h = self.dropout(h)
        h = self.linear3(h)

        return h


    def freezing(self, _model):
        for name, param in _model.named_parameters():
            param.requires_grad = False


    def get_embedding(self, posts, comments):
        if self.post_type == 'all':
            post_emb = self.text_embedding(posts)
        elif self.post_type == 'seg':
            post_emb = self.segment_embedding(posts, split=self.post_num)
        elif self.post_type == 'snt':
            post_emb = self.sentence_embedding(posts, sent_num=self.post_num)
        else:
            print('no post type')

        if self.comment_type == 'all':
            cmt_emb = self.text_embedding(comments)
        elif self.comment_type == 'seg':
            cmt_emb = self.segment_embedding(comments, split=self.comment_num)
        elif self.comment_type == 'snt':
            cmt_emb = self.sentence_embedding(comments, sent_num=self.comment_num)
        else:
            print('no comment type')

        emb = torch.cat((post_emb, cmt_emb), dim=1)

        return emb


    def text_embedding(self, texts):
        encodings = self.bert_tokenizer(texts, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        encodings.to(device)

        with torch.no_grad():
            outputs = self.bert(**encodings)
        embeddings = outputs[1]

        return embeddings

    
    def sentence_embedding(self, texts, sent_num=10):
        all_embeddings = torch.empty((0, sent_num*384)).to(device)
        
        for text in texts:
            # split sentence
            sents = []
            mask = torch.ones(sent_num)

            doc = nlp(text)
            for sent in doc.sents:
                sents.append(sent.text)
                if len(sents) >= sent_num: break

            # get embedding - SBERT
            embeddings = self.sbert.encode(sents, convert_to_tensor=True)
            embeddings.to(device)
            embeddings = self.pos_encoder(embeddings) # len(sents) * 384
        
            # padding            
            if len(sents) < sent_num:
                num_padding = sent_num - len(sents)
                padding = torch.zeros((num_padding, 384)).to(device)
                embeddings = torch.cat((embeddings, padding))
                mask[-num_padding:] = 0

            # transformer encoding
            mask = mask.unsqueeze(0).to(device)
            embeddings = embeddings.unsqueeze(1)
            encodings = self.sent_encoder(embeddings, src_key_padding_mask=mask)


            embeddings = encodings.flatten()
            embeddings = embeddings.unsqueeze(0) # 1 * (sent_num*384)

            all_embeddings = torch.cat((all_embeddings, embeddings))

        return all_embeddings


    def segment_embedding(self, texts, split=3):
        all_embeddings = torch.empty((0, split*768)).to(device)

        for text in texts:
            # split text
            sents = []
            doc = nlp(text)
            for sent in doc.sents:
                sents.append(sent.text)

            split_text = []
            l = len(sents)
            ls = l//split

            if l < split:
                split_text = sents.copy()
            else:
                for i in range(split):
                    if i != split-1:
                        split_sent = ' '.join(sents[ls*i:ls*(i+1)])
                    else:
                        split_sent = ' '.join(sents[ls*i:])
                    split_text.append(split_sent)

            # split_text embedding
            seg_embeddings = torch.empty((0, 768)).to(device)

            for _text in split_text:
                inputs = self.roberta_tokenizer(_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
                inputs.to(device)

                with torch.no_grad():
                    outputs = self.roberta(**inputs)
                seg_embedding = outputs[1]
            
                seg_embeddings = torch.cat((seg_embeddings, seg_embedding))
                
            mask = torch.ones(split)

            if len(split_text) < split:
                num_padding = split - len(split_text)
                padding = torch.zeros((num_padding, 768)).to(device)
                seg_embeddings = torch.cat((seg_embeddings, padding))
                mask[-num_padding:] = 0

            # transformer encoding
            mask = mask.unsqueeze(0).to(device)
            seg_embeddings = seg_embeddings.unsqueeze(1)
            encodings = self.seg_encoder(seg_embeddings, src_key_padding_mask=mask)


            seg_embeddings = encodings.flatten() # split*768
            seg_embeddings = seg_embeddings.unsqueeze(0) # 1 * (split*768)

            all_embeddings = torch.cat((all_embeddings, seg_embeddings))

        return all_embeddings



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


def cal_in_size(t_type, p_sent, c_sent, p_split, c_split):
    result = 0

    if t_type[0] == 'all':
        result += 768
    elif t_type[0] == 'seg':
        result += p_split*768
    elif t_type[0] == 'snt':
        result += p_sent*384

    if t_type[1] == 'all':
        result += 768
    elif t_type[1] == 'seg':
        result += c_split*768
    elif t_type[1] == 'snt':
        result += c_sent*384

    return result


