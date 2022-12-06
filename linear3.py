import snt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model = snt.s_model(768,512,3)
model.load_state_dict(torch.load('snt_model.pt'))
model.to(device)

basedata = pd.read_csv('basedata.csv')
_, test_data = train_test_split(basedata, random_state=99, test_size=0.2)

size = len(test_data)
acc = 0

for i in range(len(test_data)):
    # split sentence
    text = test_data.iloc[i,3]
    is_score = test_data.iloc[i, 4]
    es_score = test_data.iloc[i, 5]

    snt_data = pd.DataFrame(columns=['snt', 'is', 'es'])

    doc = snt.nlp(text)
    for sent in doc.sents:
        n = len(snt_data)
        snt_data.loc[n] = {'snt': sent.text, 'is': is_score-1, 'es': es_score-1}
   
    
    # embedding
    with torch.no_grad():
        encoding = snt.tokenizer(snt_data['snt'].tolist(), return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        encoding.to(device)
        output = snt.roberta(**encoding)
        embedding = output[1]
        

    # each sentence's is_score predict
    model.eval()
    with torch.no_grad():
        embeddings = embedding.to(device)
        labels = torch.tensor(snt_data['is'], device=device)

        outputs = model(embeddings)


    # get element-wise avg.
    predict = torch.mean(outputs, axis=0).argmax().item()
    if predict == is_score-1:
        acc += 1
    print('-----{}th text-----'.format(i+1))
    print('predict :', predict)
    print('is score:',is_score-1)

acc /= size
print('accuracy: {:.2f}%'.format(acc*100))
