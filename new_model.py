import torch
import torch.nn as nn
from torch.nn import functional as F
from dgl.data.utils import load_graphs

class Model(nn.Module):
    def __init__(self, d_model=768, nhead=4, dim_feedforward=512, output_dim=3, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
        self.linear3 = nn.Linear(d_model, output_dim)

    def forward(self, query, key, value):
        h, _ = self.attn(query, key, value)
        h = self.dropout1(h)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout2(h)
        h = self.norm(h)
        h = self.linear3(h)
        h = torch.squeeze(h)

        return h



def train(model, Q, K, V, labels):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        logits = model(Q, K, V)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch#{:05d}  Loss: {:.4f}'.format(epoch, loss.item()))

def test(model, Q, K, V, labels):
    model.eval()
    with torch.no_grad():
        logits = model(Q, K, V)
        labels = torch.argmax(labels, dim=1)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        print(correct.item()*1.0/len(labels))



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


is_label = test_data['is_score']

label2idx = {label:idx for idx, label in enumerate(is_label.unique())}

def one_hot_encoding(label, label2idx):
    one_hot_vector = [0]*(len(label2idx))
    idx = label2idx[label]
    one_hot_vector[idx] = 1
    return one_hot_vector

is_label_v = [one_hot_encoding(label, label2idx) for label in is_label]
is_label_v = torch.tensor(is_label_v).to(device)


Q = Q.to(device)
K = K.to(device)
V = V.to(device)

model = Model()
model.to(device)

test(model, Q, K, V, is_label_v)
