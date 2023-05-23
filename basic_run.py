# run file for basic model
""" Job script example for running in terminal
-----------------------------------------------
#!/bin/bash
mkdir ./basic/
python3 basic.py > ./basic/stdout
-----------------------------------------------
"""

import Module as M
import torch
from torch import nn
import os
from time import time
import matplotlib.pyplot as plt
#torch.set_num_threads(8)


#----- Model initialize
device='cuda' if torch.cuda.is_available() else 'cpu'
model=M.ResNet(M.BasicBlock,[2,2,2,2]).to(device)  #layers [2,2,2,2]

#--------------- Hyper parameters -----------------------------
batch_size=2
learning_rate = 1e-3
epochs = 4000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#--------------------------------------------------------------

#----- Directory for saving models during learning.
path="./basic/saved_models"
figpath="./basic/loss.png"

isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

#----- Data load --------------------
datapath="./test_lhc"
locals()

train_set=M.CustomDataset(path=datapath, train=True)
train_dataloader=M.DataLoader(train_set, batch_size=batch_size, shuffle=True)

valid_set=M.CustomDataset(path=datapath, valid=True)
valid_dataloader=M.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

#test_set=M.CustomDataset(path=datapath, test=True)
#test_dataloader=M.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#----- Save loss  --------
ylist=[]
train_loss_list=[]
valid_loss_list=[]


tic0 = time()  # For total time cal.

for t in range(epochs):

    print(f"Epoch {t + 1}\n-------------------------------")

    tic = time()

    loss_fn = nn.MSELoss()
    train_loss = M.train_loop(train_dataloader, model, loss_fn, optimizer)
    valid_loss = M.valid_loop(valid_dataloader, model, loss_fn)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    # -----------------model save---------------------
    if (t+1) % 50 == 0:
        fname = str(t + 1) + ".pt"
        PATH = os.path.join(path, fname)
        torch.save(model.state_dict(), PATH)
        print("{}-th model saved".format(t + 1))
    # ------------------------------------------------

    toc = time()

    print("time elapsed:", toc - tic)
    print()

print("Done!")

toc0 = time()

print("Total time elapsed:", toc0 - tic0)

# ---- plot the learning curve -----------
plt.plot(train_loss_list[0:epochs],label="train_loss")
plt.plot(valid_loss_list[0:epochs],label="valid_loss")
plt.legend()
plt.savefig(figpath)

