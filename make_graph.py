import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel
import dgl
from dgl.data.utils import save_graphs

basedata = pd.read_csv('./basedata.csv')
graph_data, _ = train_test_split(basedata, test_size=0.2, random_state=99)

key2idx = {key:idx for idx, key in enumerate(basedata['comment_key'])}

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(graph_data['comment_text'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

sim = []
for i in range(len(cosine_sim)):
    for j in range(i+1, len(cosine_sim[i])):
        sim.append([i, j, cosine_sim[i][j]])

sim = pd.DataFrame(sim, columns=['src', 'dst', 'sim'])
sim = sim.drop(sim[sim['sim']==0].index)

edge = sim[['src','dst']]
edge_feature = sim[['sim']]
node = graph_data.index
node_is_label = graph_data['is_score']
node_es_label = graph_data['es_score']

label2idx = {label:idx for idx, label in enumerate(graph_data['is_score'].unique())}

def one_hot_encoding(label, label2idx):
    one_hot_vector = [0]*(len(label2idx))
    idx = label2idx[label]
    one_hot_vector[idx] = 1
    return one_hot_vector

node_is_label_v = [one_hot_encoding(label, label2idx) for label in graph_data['is_score']]
node_es_label_v = [one_hot_encoding(label, label2idx) for label in graph_data['es_score']]

"""
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model.load_state_dict(torch.load('./robert.pt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

texts = graph_data['comment_text'].tolist()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
encodings = tokenizer(texts, truncation=True, return_tensors='pt', padding='max_length', max_length=512)
encodings.to(device)

with torch.no_grad():
    outputs = model(**encodings, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    first_token_tensor = last_hidden_states[:,0]
    
node_feature = first_token_tensor
"""

import torch
from transformers import RobertaTokenizer, RobertaModel, logging

logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
model.to(device)

texts = graph_data['comment_text'].tolist()
encodings = tokenizer(texts, truncation=True, return_tensors='pt', padding='max_length', max_length=512)
encodings.to(device)

with torch.no_grad():
    outputs = model(**encodings)
    #node_feature = outputs[0][:,0]
    node_feature = outputs[1]

src = torch.tensor(edge['src'].values)
dst = torch.tensor(edge['dst'].values)
g = dgl.graph((src, dst))
g = g.to(device)

g.ndata['feat'] = node_feature
g.ndata['is_label'] = torch.tensor(node_is_label.values).to(device)
g.ndata['es_label'] = torch.tensor(node_es_label.values).to(device)
g.ndata['is_label_v'] = torch.tensor(node_is_label_v, dtype=torch.float32).to(device)
g.ndata['es_label_v'] = torch.tensor(node_es_label_v, dtype=torch.float32).to(device)
g.edata['feat'] = torch.tensor(edge_feature.values).to(device)

save_graphs('./graph.bin', g)
