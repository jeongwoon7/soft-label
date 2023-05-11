import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
import Module as M
import numpy as np
from time import time
import os
#torch.set_num_threads(8)
torch.set_printoptions(precision=6)

N=4000 # number of training samples
num_qd = 5 # number of quantum dots in a system
"""
[Custom dataset for data loading]
    Each data in train, valid, test sets consists of figure (*.png) and label (*.txt).
    A figure is a simplified image of Quantum dots.
    A label is a transmission probability (as a function of energy) expressed by 1000 numerical data points.
"""

class CustomDataset(Dataset):
    def __init__(self, csv_file, path):
        self.data = pd.read_csv(csv_file, header = None, sep = " ")
        self.path = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx]).float()
        label_path = self.path + str(idx)+".txt"
        data = []
        with open(label_path, "r") as f:
            for line in f:
                a = np.float64(line.strip())
                data.append(a)
                TE = np.array(data)
        label = TE
        return sample, label


class CustomDataset2(Dataset):
    def __init__(self, csv_file, path, N=0):
        self.data = pd.read_csv(csv_file, header = None, sep = " ")
        self.path = path
        self.numtrain = N

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx]).float()
        label_path = self.path + str(idx + self.numtrain)+".txt"
        data = []
        with open(label_path, "r") as f:
            for line in f:
                a = np.float64(line.strip())
                data.append(a)
                TE = np.array(data)
        label = TE
        return sample, label

# Feed Forward Neural Network (FFNN) model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_qd, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1000)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Model initialize
device='cuda' if torch.cuda.is_available() else 'cpu'
model=NeuralNetwork().to(device)
#print(model)

#--------------- Hyper parameters -----------------------------
epochs=3000
learning_rate = 1e-3
batch_size=2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#--------------------------------------------------------------


#----- Data load --------------------
train_dataset=CustomDataset('./samples/dlists.txt','./samples/label/train/')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset=CustomDataset2('./samples/dlists2.txt','./samples/label/valid/',N=N)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

train_loss_list=[]
valid_loss_list=[]

savepath='./FNN_model/saved_models'

isExist = os.path.exists(savepath)
if not isExist:
    os.makedirs(savepath)

for t in range(epochs):

    print(f"Epoch {t + 1}\n-------------------------------")

    tic = time()

    loss_fn = nn.MSELoss()
    train_loss = M.train_loop(train_dataloader, model, loss_fn, optimizer)
    valid_loss = M.valid_loop(valid_dataloader, model, loss_fn)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    # -----------------model save---------------------
    if (t+1) % 1000 == 0:
        fname = str(t + 1) + ".pt"
        PATH = os.path.join(savepath, fname)
        torch.save(model.state_dict(), PATH)
        print("{}-th model saved".format(t + 1))
    # ------------------------------------------------

    toc = time()

    print("time elapsed:", toc - tic)
    print()

print("Done!")
