from transformers import BertTokenizer, BertModel, logging, AdamW, AutoModel
import torch
import torch.nn as nn
from torch import Tensor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import spacy
from sklearn.metrics import f1_score
from spacy.lang.en import English
import gc
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import math
import warnings
from captum.attr import IntegratedGradients
from textsplit import text_segmentation

warnings.simplefilter(action='ignore', category=FutureWarning)

gc.collect()
torch.cuda.empty_cache()

score_type = 'is_score'
basedata = pd.read_csv('./basedata.csv')
data = basedata[['post_text', 'comment_text', score_type]]

logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlp = English()
nlp.add_pipe("sentencizer")


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

class model(nn.Module):
    def __init__(self, num_seg=None, num_class=3, d_model=768, dropout=0.1, device='cuda'):
        super().__init__()

        self.device = device
        self.d_model = d_model
        
        if self.d_model == 768:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.h_size = 384
        elif self.d_model == 384:
            self.bert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.h_size = 256
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.positional_encoder = PositionalEncoding(self.d_model)

        self.num_seg = num_seg
        self.num_class = num_class
        self.droout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
                nn.Linear(self.d_model, self.h_size),
                nn.ReLU(),
                #nn.Linear(self.h_size, self.h_size),
                #nn.ReLU(),
                nn.Linear(self.h_size, self.num_class),
                )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_masks):
        outputs = torch.empty((0, self.num_class)).to(device)

        for ids, mask in zip(input_ids, attention_masks):
            ids = ids.to(torch.long)
            output = self.bert(input_ids=ids, attention_mask=mask)
            output = self.positional_encoder(output[1])

            #mask, src_key_padding_mask = self.make_mask(len(ids), mask)

            #output = self.encoder(output, mask=mask, src_key_padding_mask=src_key_padding_mask)
            output = self.encoder(output)
            output = self.classifier(output)
            output = self.softmax(output)
            output = output.sum(dim=0).unsqueeze(0)

            outputs = torch.cat((outputs, output))
        return outputs

    def freezing(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def make_mask(self, max_seg, attention_masks):
        mask = (torch.triu(torch.ones(max_seg, max_seg))==1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0)).to(self.device)

        src_key_padding_mask = (attention_masks==0).all(dim=1)

        return mask, src_key_padding_mask





class Dataset(torch.utils.data.Dataset):
    def __init__(self, posts, comments, labels, limit=(None, None)):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.post_texts = posts
        self.comment_texts = comments

        self.post_input_ids, self.post_attention_masks = self.get_sequence_encoding(posts, limit[0])
        self.comment_input_ids, self.comment_attention_masks = self.get_sequence_encoding(comments, limit[1])
        self.labels = labels

    def __getitem__(self, idx):
        item = {'post_input_ids': self.post_input_ids[idx], 
                'post_attention_masks': self.post_attention_masks[idx], 
                'post_texts': self.post_texts[idx],
                'comment_input_ids': self.comment_input_ids[idx], 
                'comment_attention_masks': self.comment_attention_masks[idx],
                'comment_texts': self.comment_texts[idx]
                }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def get_sequence_encoding(self, contents, limit=None):
        # get sequence
        sequences = []
        max_num = 0

        for content in contents:
            sentences = list(map(lambda x: str(x), list(nlp(content).sents)))
            segments = text_segmentation(sentences)
            segments = list(map(lambda x: x.replace('\"', '\'').replace(' -- ', ' '), segments))

            if limit and len(segments) > limit:
                segments = segments[:limit]

            sequences.append(segments)

            if max_num < len(segments):
                max_num = len(segments)
        
        # get encoding
        total_input_ids = []
        total_attention_masks = []

        for seq in sequences:
            result = self.tokenizer(seq, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
            
            input_ids = result['input_ids']
            attention_masks = result['attention_mask']

            pad = (0, 0, 0, max_num-len(input_ids))
            input_ids = nn.functional.pad(input_ids, pad, 'constant', 0).to(device)
            attention_masks = nn.functional.pad(attention_masks, pad, 'constant', 0).to(device)

            total_input_ids.append(input_ids)
            total_attention_masks.append(attention_masks)

        total_input_ids = torch.stack(total_input_ids, dim=0)
        total_attention_masks = torch.stack(total_attention_masks, dim=0)

        return total_input_ids, total_attention_masks

def get_segments_with_padding(text, limit=None):
    sentences = list(map(lambda x: str(x), list(nlp(text).sents)))
    segments = text_segmentation(sentences)
    segments = list(map(lambda x: x.replace('\"', '\'').replace(' -- ', ' '), segments))

    if limit and len(segments) > limit:
        segments = segments[:limit]
    elif limit and len(segments) < limit:
        for i in range(limit-len(segments)):
            segments.append('padding'+str(i+1))

    return segments

def custom_forward(embeddings):
    output = model.positional_encoder(embeddings)
    output = model.encoder(output)
    output = model.classifier(output)
    output = model.softmax(output)

    output = output.sum(dim=0).unsqueeze(0)
    # 1 * 3
    return output



model = model(d_model=768, device=device)
model.load_state_dict(torch.load('./model/final_seg/model.pt'))
model.to(device)
model.eval()

batch_size = 1 

limit = (3, 3)

data = data[:20]

dataset = Dataset(data['post_text'].tolist(), data['comment_text'].tolist(), (data[score_type]-1).tolist(), limit=limit)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

ig = IntegratedGradients(custom_forward)


data_post = []
data_comment = []
data_score = []
data_predict = []
data_ig_segment = []
data_ig_segment_type = []

with torch.no_grad():
    for batch in dataloader:
        label = batch['labels'].to(device)
        
        post_input_ids = batch['post_input_ids']
        post_attention_masks = batch['post_attention_masks']

        comment_input_ids = batch['comment_input_ids']
        comment_attention_masks = batch['comment_attention_masks']

        input_ids = torch.cat((post_input_ids, comment_input_ids), dim=1)
        input_ids.to(device)
        attention_masks = torch.cat((post_attention_masks, comment_attention_masks), dim=1)
        attention_masks.to(device)

        model.zero_grad()

        pred = model(input_ids, attention_masks)
        pred = pred.softmax(1).argmax(1)

        #print('Predict:', pred.item())
        #print('Label  :', label.item())
        data_score.append(label.item())
        data_predict.append(pred.item())

        ig_input = model.bert(input_ids=input_ids[0], attention_mask=attention_masks[0])
        ig_input = ig_input[1]

        attributions = ig.attribute(inputs=ig_input, target=pred)

        #print(attributions.size()) # 6 * 384
        att = torch.abs(attributions)
        seg_score = att.sum(dim=-1).squeeze()
        idx = torch.argmax(seg_score)
        seg_type = 'post' if idx < limit[0] else 'comment'

        post_text = batch['post_texts'][0]
        post_segments = get_segments_with_padding(post_text, limit[0])
        comment_text = batch['comment_texts'][0]
        comment_segments = get_segments_with_padding(comment_text, limit[1])
        segments = post_segments + comment_segments

        #print('POST')
        #print(post_text)
        data_post.append(post_text)
        #print()
        #print('COMMENT')
        #print(comment_text)
        data_comment.append(comment_text)
        #print()
        #print('IG segment ({})'.format(seg_type))
        data_ig_segment_type.append(seg_type)
        #print(segments[idx])
        data_ig_segment.append(segments[idx])
        #print('='*100)



ig_df = pd.DataFrame({'post_text': data_post, 'comment_text': data_comment,
                        'score': data_score, 'predict': data_predict,
                        'ig_segment': data_ig_segment, 'type': data_ig_segment_type})

print(ig_df)

#ig_df.to_csv('./bert_p3_c3_65_ig.csv', index=False)
