from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, logging
import torch
import pandas as pd
import matplotlib.pyplot as plt

basedata = pd.read_csv('./basedata.csv')
_, test_data = train_test_split(basedata, test_size=0.2, random_state=99)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.set_verbosity_error()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.to(device)

encodings = tokenizer(test_data['comment_text'].tolist(), return_tensors='pt', truncation=True, max_length=512, padding='max_length')
encodings.to(device)

with torch.no_grad():
    outputs = roberta(**encodings)
    vectors = outputs[1]

is_score = test_data['is_score'].tolist()
es_score = test_data['es_score'].tolist()

model = TSNE()

tsne = model.fit_transform(vectors.cpu())

plt.subplot(121)
plt.scatter(tsne[:,0], tsne[:,1], 25, is_score)
plt.colorbar()
plt.title('is_score')

plt.subplot(122)
plt.scatter(tsne[:,0], tsne[:,1], 25, es_score)
plt.colorbar()
plt.title('es_score')

plt.tight_layout()
plt.savefig('tsne.jpg')
