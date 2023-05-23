# run file for loss prediction model
import Module as M
import torch
from torch import nn
import os
from time import time
import matplotlib.pyplot as plt

from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import random

trans=transforms.Compose([transforms.ToTensor()])
torch.set_num_threads(8)

#----- Model initialize
device='cuda' if torch.cuda.is_available() else 'cpu'
model=M.ResNet(M.BasicBlock,[2,2,2,2]).to(device)  #layers
model2=M.loss_pred_ResNet(M.BasicBlock,[2,2,2,2]).to(device)

""" --------------------------------------------------------------------------------------------------------
- num_add = num_add_half * 2
- Add 100 samples per {freq} epochs.  
    - sample 50 based on the predicted loss of the loss prediction model
    - sample 50 for random
- Initially, train, test, and valid set consist of 200, 199600, and 200 data points. (total, 200000 in the original work))
- Here, "test" set is actually "candidate" set; 
  data with the large predicted loss will be added to train set during the learning.
- A separate test data should be used for analysis with a trained model. 
- Index for data files (in the original work)
    train set : from 0 to 199
    test set : from 200 to 199799
    valid set : from 199800 to 199999 
    
- Due to the storage limit, we counldn't upload the Data directory "test-200000" with 200000 data.
    * However, to see how the adaptive run work, you can use "test" with 4000 data, instead. 
    * in this case, train set : from 0 to 199, test set : from 200 to 3799, and valid set : from 3800 to 3999
-------------------------------------------------------------------------------------------------------------
"""
freq = 100 # add samples per freq.
num_add_half=50
Ni=200
Nf=3800  #199800
test_sample_list=[a for a in range(Ni,Nf)]

#--------------- Hyper parameters -----------------------------
batch_size=2
learning_rate = 1e-3
epochs = 2500

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate)

#------------- Data load --------------------
datapath="./test/"  # upper directory of train, test, valid directories
savedir="./ML_adapt/" # directory for saving models during learning
#--------------------------------------------------------------

path=savedir+"saved_models"
figpath=savedir+"/loss.png"

isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

# ----- save loss -------
train_loss_list = []
train_loss2_list = []
valid_loss_list = []

# ---- loss_predict
tic0=time()  # For total time cal.

def loss_predict(test_sample_list):
    loss_list=[]

    # data file name :  "i.png", "i.txt"
    for i in test_sample_list :
        fname=str(i)+".png"
        fname2=str(i)+".txt"
        p1=datapath+"figures/test/"
        img_path=os.path.join(p1,fname) # Actually, not a test set but a candidate data set to add to the train set.

        img=Image.open(img_path).convert("L") # greyscale
        img=trans(img)
        img.unsqueeze(0).shape

        pred2=model2(img.unsqueeze(0).to(device))
        y2_pred=pred2.squeeze(0).detach().cpu().numpy()
        rs_pred=np.sqrt(y2_pred[0]**2)
        loss_list.append((i,rs_pred))

    check=len(loss_list)

    if check :
        sorted_idx=np.array(loss_list)[:,1].argsort(axis=-1)[::-1]

    # Adaptive sampling procedure
    # Predict loss of the data in candidate set (though, the directory name is "test").
    # Based on the prediction of model2, move data with the large losses in the candidate set to the train set.

    list_to_remove = [loss_list[a][0] for a in sorted_idx[0:num_add_half]]

    return list_to_remove

# learning
for t in range(epochs):

    locals()
    train_set=M.CustomDataset(path=datapath, train=True)
    train_dataloader=M.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    valid_set=M.CustomDataset(path=datapath, valid=True)
    valid_dataloader=M.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    test_set=M.CustomDataset(path=datapath, test=True)
    test_dataloader=M.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    print(f"Epoch {t + 1}\n-------------------------------")

    tic = time()

    loss_fn = nn.MSELoss()
    loss_fn2 = nn.MSELoss()
    #train_loss = M.train_loop(model2, loss_fn2, optimizer2r)
    train_loss,train_loss2=M.train_loop2(train_dataloader, model, loss_fn, optimizer, model2, loss_fn2, optimizer2)
    valid_loss = M.valid_loop(valid_dataloader, model, loss_fn)

    train_loss_list.append(train_loss)
    train_loss2_list.append(train_loss2)
    valid_loss_list.append(valid_loss)

    # ----------------- model save & sample ---------------------
    if (t+1) % freq == 0:
        fname = str(t + 1) + ".pt"
        PATH = os.path.join(path, fname)
        torch.save(model.state_dict(), PATH)
        fname2 = str(t + 1)+"-model2.pt"
        PATH2 = os.path.join(path, fname2)
        torch.save(model2.state_dict(), PATH2)

        print("{}-th model saved".format(t + 1))


        """ 
        Add samples from candidates based on the loss
        """
        adaptive_sample = loss_predict(test_sample_list)
        dummy = [i for i in test_sample_list if i not in adaptive_sample]
        random_sample = random.sample(dummy,num_add_half)
        data_to_remove = adaptive_sample + random_sample

        print(" -- Sampled data --")
        print(data_to_remove)

        for s in data_to_remove:
            fname = str(s) + ".png"
            fname2 = str(s) + ".txt"
            os.rename(datapath + '/figures/test/' + fname, datapath + '/figures/train/' + fname)
            os.rename(datapath + '/label/test/' + fname2, datapath + '/label/train/' + fname2)

        for s in data_to_remove:
            test_sample_list.remove(s)
    # ------------------------------------------------

    toc = time()

    print("time elapsed:", toc - tic)
    print()

    # sample redistribute

print("Done!")

toc0 = time()
print("Total time elapsed:", toc0 - tic0)

# ----- plot the learning curve -----------
plt.plot(train_loss_list[0:epochs],label="train_loss")
plt.plot(train_loss2_list[0:epochs],label="train2_loss")
plt.plot(valid_loss_list[0:epochs],label="valid_loss")
plt.legend()
#plt.show()
plt.savefig(figpath)

