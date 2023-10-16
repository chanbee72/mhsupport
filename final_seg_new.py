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
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import math
import warnings
from textsplit import text_segmentation
warnings.simplefilter(action='ignore', category=FutureWarning)

score_type = 'is_score'
basedata = pd.read_csv('./basedata.csv')
data = basedata[['post_text', 'comment_text', score_type]]

train_idx, test_idx = train_test_split(data.index.tolist(), test_size=0.2, random_state=99)
train_data = data.iloc[train_idx]
test_data = data.iloc[test_idx]

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
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.h_size = 512
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.positional_encoder = PositionalEncoding(self.d_model)

        self.num_seg = num_seg
        self.num_class = num_class
        self.classifier = nn.Sequential(
                nn.Linear(self.d_model*self.num_seg, self.h_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.h_size, self.num_class),
                )

    def forward(self, input_ids, attention_masks):
        outputs = torch.empty((0, self.num_class)).to(device)

        for ids, att_mask in zip(input_ids, attention_masks):
            output = self.bert(input_ids=ids, attention_mask=att_mask)
            output = self.positional_encoder(output[1])
            
            mask, src_key_padding_mask = self.make_mask(len(ids), att_mask)

            #output = self.encoder(output, mask=mask, src_key_padding_mask=src_key_padding_mask)
            output = self.encoder(output, src_key_padding_mask=src_key_padding_mask)
            #output = self.encoder(output)

            output = torch.transpose(output, 0, 1)

            diag_mask = self.make_diagonal(src_key_padding_mask)
            output = torch.matmul(output, diag_mask)
            output = torch.flatten(output)
            
            output = self.classifier(output)
            output = output.unsqueeze(0)

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

    def make_diagonal(self, padding_mask):
        matrix = torch.diag(~padding_mask).float()

        return matrix





class Dataset(torch.utils.data.Dataset):
    def __init__(self, posts, comments, labels, limit=(None, None)):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.post_input_ids, self.post_attention_masks = self.get_sequence_encoding(posts, limit[0])
        self.comment_input_ids, self.comment_attention_masks = self.get_sequence_encoding(comments, limit[1])
        self.labels = labels

    def __getitem__(self, idx):
        item = {'post_input_ids': self.post_input_ids[idx], 
                'post_attention_masks': self.post_attention_masks[idx], 
                'comment_input_ids': self.comment_input_ids[idx], 
                'comment_attention_masks': self.comment_attention_masks[idx]}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def get_sequence_encoding(self, contents, limit=None):
        # get sequence
        sequences = []
        max_num = limit

        for content in contents:
            sentences = list(map(lambda x: str(x), list(nlp(content).sents)))
            segments = text_segmentation(sentences)
            segments = list(map(lambda x: x.replace('\"', '\'').replace(' -- ', ' '), segments))

            if limit and len(segments) > limit:
                segments = segments[:limit]

            sequences.append(segments)

        
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

limit = (10, 5)

model = model(num_seg=sum(limit), d_model=768, device=device)
model.to(device)

batch_size = 4
num_epochs = 10
learning_rate = 1e-5

train_dataset = Dataset(train_data['post_text'].tolist(), train_data['comment_text'].tolist(), (train_data[score_type]-1).tolist(), limit=limit)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = Dataset(test_data['post_text'].tolist(), test_data['comment_text'].tolist(), (test_data[score_type]-1).tolist(), limit=limit)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
loss_fn = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=2, gamma=0.8) 

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='macro'), f1_score(labels_flat, preds_flat, average='micro')

def accuracy_per_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class {label} : ', end='')
        print(f'{len(y_preds[y_preds == label])}/{len(y_true)}')

def train(dataloader, model, optimizer):
    model.train()

    epoch_loss = 0
    size = len(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        labels = batch['labels'].to(device)

        post_input_ids = batch['post_input_ids']
        post_attention_masks = batch['post_attention_masks']
        comment_input_ids = batch['comment_input_ids']
        comment_attention_masks = batch['comment_attention_masks']

        input_ids = torch.cat((post_input_ids, comment_input_ids), dim=1)
        input_ids.to(device)
        attention_masks = torch.cat((post_attention_masks, comment_attention_masks), dim=1)
        attention_masks.to(device)

        outputs = model(input_ids, attention_masks)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    scheduler.step()

    print('Training Loss: {:.3f}'.format(epoch_loss/size))


def test(dataloader, model):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, accuracy = 0, 0

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)

            post_input_ids = batch['post_input_ids']
            post_attention_masks = batch['post_attention_masks']
            comment_input_ids = batch['comment_input_ids']
            comment_attention_masks = batch['comment_attention_masks']

            input_ids = torch.cat((post_input_ids, comment_input_ids), dim=1)
            input_ids.to(device)
            attention_masks = torch.cat((post_attention_masks, comment_attention_masks), dim=1)
            attention_masks.to(device)

            pred = model(input_ids, attention_masks)

            loss = loss_fn(pred, labels)
            test_loss += loss.item()
            accuracy += (pred.softmax(1).argmax(1) == labels).type(torch.float).sum().item()

            predictions.append(pred.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())

    test_loss /= size
    accuracy /= size

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    accuracy_per_class(predictions, true_labels)
    test_f1_macro, test_f1_micro = f1_score_func(predictions, true_labels)

    print("Test Loss: {:.3f}, Accuracy: {:.2f}%".format(test_loss, accuracy*100))
    print("F1 Score(macro): {:.4f}, F1 Score(micro): {:.4f}".format(test_f1_macro, test_f1_micro))

    return accuracy*100

from tqdm import tqdm

for epoch in tqdm(range(1, num_epochs+1)):
    print("Epoch #{}".format(epoch))
    train(train_loader, model, optimizer)
    acc = test(test_loader, model)

    gc.collect()
    torch.cuda.empty_cache()

#if acc > 65:
#    torch.save(model.state_dict(), './model/final_seg_new/model_{}.pt'.format(int(acc)))
#    print('Save model')
