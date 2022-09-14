import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data.utils import load_graphs
from sklearn.metrics import *

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

def performance(y, pred):
    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    #roc_auc = roc_auc_score(y, pred)
    print('Accuracy: {:.4f} | F1 score: {:.4f}'.format(accuracy, f1))


graph = load_graphs('./graph.bin')[0][0]

idx = [i for i in range(graph.num_nodes())]
train_idx = idx[:int(len(idx)*0.8)]
valid_idx = idx[int(len(idx)*0.8):]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
graph = graph.to(device)
train_idx = torch.tensor(train_idx).to(device)
valid_idx = torch.tensor(valid_idx).to(device)

model = SAGE(graph.ndata['feat'].shape[1], 256, 3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

sampler = dgl.dataloading.NeighborSampler([15,10,5], prefetch_node_feats=['feat'], prefetch_labels=['is_label_v'])
train_dataloader = dgl.dataloading.DataLoader(graph, train_idx, sampler, device=device, batch_size=16, shuffle=True, drop_last=False, num_workers=0, use_uva=False)
valid_dataloader = dgl.dataloading.DataLoader(graph, valid_idx, sampler, device=device, batch_size=16, shuffle=True, drop_last=False, num_workers=0, use_uva=False)

for _ in range(100):
    model.train()
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        x = blocks[0].srcdata['feat']
        y = blocks[-1].dstdata['is_label_v']
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 20 == 0:
            acc = MF.accuracy(torch.argmax(y_hat, dim=1), torch.argmax(y, dim=1))
            print('Loss: {:.4f} | Acc: {:.4f}'.format(loss.item(), acc.item()))

    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(torch.argmax(blocks[-1].dstdata['is_label_v'], dim=1))
            y_hats.append(torch.argmax(model(blocks, x), dim=1))
    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))
    print('Validation acc: {:.4f}'.format(acc.item()))
