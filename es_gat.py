import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data.utils import load_graphs

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[0], out_size, heads[1], activation=None))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = torch.argmax(labels[mask], dim=1)
        _, indices = torch.max(logits, dim=1) 
        correct = torch.sum(indices == labels)
        return correct.item()*1.0/len(labels)

def train(g, features, labels, masks, model):
    train_mask = masks[0]
    valid_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-4)

    for epoch in range(2000):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc= evaluate(g, features, labels, valid_mask, model)
        if epoch % 100 == 0 or epoch == 1999:
            print('Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}'.format(epoch, loss.item(), acc))

g = load_graphs('./graph.bin')[0][0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = g.int().to(device)
g = dgl.add_self_loop(g)

features = g.ndata['feat']
labels = g.ndata['es_label_v']
train_mask = [True if i < int(g.num_nodes()*0.8) else False for i in range(g.num_nodes())]
valid_mask = [True if i >= int(g.num_nodes()*0.8) else False for i in range(g.num_nodes())]
masks = train_mask, valid_mask

in_size = features.shape[1]
out_size = 3
model = GAT(in_size, 8, out_size, heads=[8,1]).to(device)

print('Training...')
train(g, features, labels, masks, model)
