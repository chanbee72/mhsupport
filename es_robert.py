import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, logging
import torch
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


basedata = pd.read_csv('./basedata.csv')
data = basedata[['post_text', 'comment_text', 'es_score']]

train_idx, valid_idx = train_test_split(data.index.tolist(), test_size=0.2, random_state=99)
train_data = data.iloc[train_idx]
valid_data = data.iloc[valid_idx]


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings = tokenizer(train_data['post_text'].tolist(), train_data['comment_text'].tolist(), truncation=True, padding=True)
train_labels = (train_data['es_score']-1).tolist()
valid_encodings = tokenizer(valid_data['post_text'].tolist(), valid_data['comment_text'].tolist(), truncation=True, padding=True)
valid_labels = (valid_data['es_score']-1).tolist()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


num_of_epochs = 10
learning_rate = 2e-5

train_dataset = Dataset(train_encodings, train_labels)
valid_dataset = Dataset(valid_encodings, valid_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)


logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)


def train(dataloader, model, optimizer):
    model.train()

    epoch_loss = 0
    size = len(dataloader.dataset)

    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        optimizer.zero_grad()
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('Training loss: {:.3f}'.format(epoch_loss/size))


def test(dataloader, model):
    model.eval()

    size = len(dataloader.dataset)
    test_loss, accuracy = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['input_ids'].to(device), batch['labels'].to(device)
            pred = model(x, labels=y)

            test_loss += pred.loss
            accuracy += (pred.logits.softmax(1).argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        accuracy /= size
        
        print('Test loss: {:.3f}, Accuracy: {:.3f}%'.format(test_loss, accuracy*100))


from tqdm.auto import tqdm

tqdm.pandas()

for i in tqdm(range(num_of_epochs)):
    print("Epoch: #{}".format(i+1))
    train(train_loader, model, optimizer)
    test(valid_loader, model)

#torch.save(model.state_dict(), './robert.pt')
