from util import *
import time
import torch

import networkx as nx
from typing import Union, Tuple

from utils import *
from config import emb_config

random.seed(0)

runs = 1
epochs = 2000
lr = 0.01
weight_decay = 5e-5
early_stopping = 0
hidden = 64
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_base = os.getcwd()
input_dim = 16
emb_dim = 8
num_hidden = 0
dir_name = "output"

class LossFunc(torch.nn.Module, sigma):
    def __init__(self, weight=None, size_average=None):
        super(LossFunc, self).__init__()

    def forward(self, output, target, sigma): 
        loss_m = matmul(output, transpose(target, 0, 1))
        loss_m -= torch.diag(loss_m) if np.dot(output,transpose(target, 0, 1))/(norm(output)*norm(transpose(target, 0, 1)))>sigma else 0
        loss = torch.mean(sigmoid(F.normalize(loss_m, dim=0)))
        return loss

class SAGE_VE(torch.nn.Module):
    def __init__(self, dataset, aggr='mean'):
        super(SAGE_VE, self).__init__()
        self.dataset = dataset
        self.num_input_features = input_dim
        self.num_hidden_features = emb_dim
        self.num_output_featrues = emb_dim
        self.num_hiddent_layers = num_hidden
        self.conv1 = SAGEConv(self.num_input_features, self.num_hidden_features)
        self.conv2 = SAGEConv(self.num_hidden_features, self.num_hidden_features)
        self.conv3 = SAGEConv(self.num_hidden_features, self.num_output_featrues)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        if self.num_hiddent_layers==0:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            for layer in range(self.num_hiddent_layers):
                x = self.conv2(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def run(self, out_fname, model, runs, epochs, lr, dropout, sigma=0.05, weight_decay=0.001, early_stopping=True):
        val_losses, durations = [], []
        for _ in range(runs):
            data = self.dataset
            data = data.to(device)

            model.to(device).reset_parameters()
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            best_loss = float('inf')
            loss_history = []

            for epoch in range(1, epochs + 1):
                train(model, optimizer, data, sigma)
                eval_info = evaluate(model, data)
                eval_info["epoch"] = epoch

                if eval_info["loss"] < best_loss:
                    best_loss = eval_info["loss"]

                loss_history.append(eval_info["loss"])
                if early_stopping > 0 and epoch > epochs // 2:
                    tmp = Tensor(loss_history[-(early_stopping + 1):-1])
                    if eval_info["loss"] > tmp.mean().item():
                        break
                print(eval_info["loss"])

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()

            val_losses.append(best_loss)
            durations.append(t_end - t_start)

        loss, duration = Tensor(val_losses), Tensor(durations)

        out = model(data)
        cur_out, _ = torch.split(out, int(out.size()[0]/2))
        cur_out = cur_out.reshape((cur_out.size()[1], cur_out.size()[2]))
        cur_out = F.normalize(cur_out, dim=0)

        f_out = open(out_fname, "w")
        f_out.write("{} {}\n".format(cur_out.size()[0], emb_dim))
        for i in range(cur_out.size()[0]):
            f_out.write(f"{data.x[i]} ")
            for j in range(emb_dim):
                f_out.write(f"{cur_out[i][j]} ")
            f_out.write("\n")
        f_out.close()

        print('Val Loss: {:.4f}, Duration: {:.3f}'.
            format(loss.mean().item(), duration.mean().item()))

def train(model, optimizer, data, sigma):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    (h, l) = torch.split(out, int(out.size()[0]/2))
    h = h.reshape((h.size()[1], h.size()[2]))
    l = l.reshape((l.size()[1], l.size()[2]))

    L = LossFunc(simga)
    loss = L(h, l)
    loss.backward()
    optimizer.step()

def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    (h, l) = torch.split(logits, int(logits.size()[0]/2))
    h = h.reshape((h.size()[1], h.size()[2]))
    l = l.reshape((l.size()[1], l.size()[2]))
    L = LossFunc()
    loss = L(h, l)
    outs["loss"] = loss

    return outs

if __name__ == "__main__":
    cur_g = SAGE_VE(get_data("output/GF/raw_data/GF.edgelist")[1])
    cur_g.run(runs=3, epochs=50, lr=0.01, dropout=0.5, out_fname="output/GF/output/cur_out.txt")