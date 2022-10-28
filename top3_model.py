import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import RobertaModel, RobertaTokenizer, logging
from torch.utils.data import DataLoader


logging.set_verbosity_error()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base').to(device)

tfidf = TfidfVectorizer()

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        #fc1
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        #fc2
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu3 = nn.ReLU()

    def forward(self, base, inputs):
        top3_vectors = self.get_top3vector(base, inputs)
        
        h = self.relu1(self.linear1(top3_vectors))
        h = self.dropout1(h)
        h = self.relu2(self.linear2(h))
        h = self.dropout2(h)
        h = self.linear3(h)
        #h = self.relu3(self.linear3(h))

        return h

    def get_top3vector(self, base_comments, input_comments):
        top3vector = []

        for i, comment in enumerate(input_comments):
            comments = base_comments.copy()
            comments.insert(0, comment)

            tfidf_matrix = tfidf.fit_transform(comments)
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            sim = torch.tensor(cosine_sim[0][1:])

            _, indices = torch.topk(sim, 3)
            top3comment = [comments[idx+1] for idx in indices]

            encodings = tokenizer(top3comment, truncation=True, return_tensors='pt', padding='max_length', max_length=512)
            encodings.to(device)

            with torch.no_grad():
                outputs = roberta(**encodings)
                vectors = outputs[1]
            
            #top3vector.append(torch.sum(vectors, axis=0).tolist())
            top3vector.append(torch.flatten(vectors).tolist())

        return torch.tensor(top3vector, device=device)


def train(model, base, dataloader):
    num_epochs = 10

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-3)

    for epoch in range(num_epochs+1):
        for batch_idx, samples in enumerate(dataloader):
            inputs, labels = samples
            
            logits = model(base, inputs)
            loss = loss_fn(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch {:3d}/{}  Batch {:3d}/{} Loss: {:.4f}'.format(epoch, num_epochs, batch_idx+1, len(dataloader), loss.item()))



def test(model, base, inputs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(base, inputs)
        predict = torch.argmax(logits, dim=1)
        correct = torch.sum(predict==labels)
        print(logits)
        print(correct.item()*1.0/len(labels))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, comments, labels):
        self.comments = comments
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        label = self.labels[idx]
        return comment, label


def label2vec(labels):
    labels_ = labels.unique()
    labels_.sort()
    dic = {key : value for value, key in enumerate(labels_)}
    
    return_vec = []

    for label in labels:
        vec = [0]*len(labels_)
        idx = dic[label]
        vec[idx] = 1
        return_vec.append(vec)
    
    return return_vec


basedata = pd.read_csv("./basedata.csv")
train_data, test_data = train_test_split(basedata, test_size=0.2, random_state=99)

model = Model(input_size=768*3, hidden_size=512, output_size=3, dropout=0.1)
model.to(device)

base = train_data['comment_text'].tolist()

comments = basedata['comment_text'].tolist()
test_comments = test_data['comment_text'].tolist()

labels = torch.tensor((basedata['is_score']-1).tolist(), device=device)
test_labels = torch.tensor((test_data['is_score']-1).tolist(), device=device)

#is_labels = torch.tensor((test_data['is_score']-1).tolist(), device=device)
#es_labels = torch.tensor((test_data['es_score']-1_.tolist(), device=device)

labels_vec = label2vec(basedata['is_score'])
labels_vec = torch.FloatTensor(labels_vec).to(device)

dataset = Dataset(comments, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print('Training ...')
train(model, base, dataloader)

print('Test ...')
test(model, base, test_comments, test_labels)
