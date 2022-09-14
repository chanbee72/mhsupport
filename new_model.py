import torch
import torch.nn as nn
from torch.nn import functional as F
from dgl.data.utils import load_graphs

class Model(nn.Module):
    def __init__(self, d_model=768, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu



    def forward(self, query, key, value):
        h, _ = self.attn(query, key, value)
        h = self.dropout1(h)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout2(h)
        h = self.norm(h)
        
        return h


graph = load_graphs('./graph.bin')[0][0]
K = graph.ndata['feat']
K = K.reshape(K.size()[0], -1, K.size()[1])
V = K


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaModel, RobertaTokenizer, logging

basedata = pd.read_csv('./basedata.csv')
_, test_data = train_test_split(basedata, test_size=0.2, random_state=99)
texts = test_data['comment_text'].tolist()

logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.to(device)

embeddings = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt').to(device)
with torch.no_grad():
    Q = roberta(**embeddings)[1]
Q = Q.reshape(Q.size()[0], -1, Q.size()[1])

print('Query: ', Q.size())
print('Key: ',K.size())
print('Value: ', V.size())

model = Model()
model.to(device)
output = model(Q.to(device), K.to(device), V.to(device))
print(output.size())
