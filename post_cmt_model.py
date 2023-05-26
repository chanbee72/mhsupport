import torch
from torch import nn, Tensor
from transformers import RobertaModel, RobertaTokenizer, logging, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import spacy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math



device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.set_verbosity_error()

nlp = spacy.load("en_core_web_sm")

basedata = pd.read_csv('./basedata.csv')

"""
def sent_num(texts):
    num_list = []
    
    for text in texts:
        doc = nlp(text)
        num = len(list(doc.sents))
        num_list.append(num)
    
    print('max num : {}'.format(max(num_list)))
    print("mean num: {}".format(sum(num_list)/len(num_list)))

print("total data: {}".format(len(basedata)))
print("---POST TEXT---")
sent_num(basedata['post_text'].tolist())
print("---COMMENT TEXT---")
sent_num(basedata['comment_text'].tolist())
"""

train, test = train_test_split(basedata, test_size=0.2, random_state=99)

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


score = 'is_score'
batch_size = 32

train_dataset = Dataset(train['post_text'].tolist(), train['comment_text'].tolist(), (train[score]-1).tolist())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataset = Dataset(test['post_text'].tolist(), test['comment_text'].tolist(), (test[score]-1).tolist())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

#print("# of train: {}".format(len(train_dataset)))
#print("# of test : {}".format(len(test_dataset)))


class model(nn.Module):
    def __init__(self, in_size, hid_size, out_size, t_type=('all', 'all', 0, 0), dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hid_size)
        self.linear2 = nn.Linear(hid_size, hid_size)
        self.linear3 = nn.Linear(hid_size, out_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta = RobertaModel.from_pretrained('roberta-base').eval()
        self.roberta.to(device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').eval()
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2').eval()
        self.sbert.to(device)

        self.post_type = t_type[0]
        self.comment_type = t_type[1]
        self.post_num = t_type[2]
        self.comment_num = t_type[3]

        self.freezing(self.roberta)
        self.freezing(self.bert)
        self.freezing(self.sbert)

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
        encodings = self.roberta_tokenizer(texts, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        encodings.to(device)

        with torch.no_grad():
            outputs = self.roberta(**encodings)
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



def train(dataloader, model, optimizer, loss_fn):
    model.train()

    epoch_loss = 0
    size = len(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        posts = batch['post']
        comments = batch['comment']
        labels = batch['label']

        outputs = model(posts, comments)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print("Train Loss: {:.6f}".format(epoch_loss/size))

def test(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, acc = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            posts = batch['post']
            comments = batch['comment']
            labels = batch['label']

            outputs = model(posts, comments)

            test_loss += loss_fn(outputs, labels).item()
            acc += (outputs.softmax(1).argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
    
    print("Test Loss : {:.6f}, Accuracy: {:.2f}".format(test_loss/size, acc*100/size))

    return acc*100/size


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


p_sent_num = 20
c_sent_num = 10
p_split = 3
c_split = 5
t_type = ('all', 'snt', p_split, c_sent_num)
in_size = cal_in_size(t_type, p_sent_num, c_sent_num, p_split, c_split)
h_size = 768
out_size = 3

model = model(in_size, h_size, out_size, t_type=t_type, dropout=0.2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 50

max_acc = 0
for i in tqdm(range(num_epochs)):
    print("Epoch {}".format(i+1))
    train(train_dataloader, model, optimizer, loss_fn)
    acc = test(test_dataloader, model, loss_fn)

    if max_acc < acc:
        max_acc = acc

print("Max Accuracy: {:.2f}".format(max_acc))
