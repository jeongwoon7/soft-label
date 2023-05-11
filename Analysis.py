import Module as M
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision.transforms import transforms

torch.set_printoptions(precision=6)
trans = transforms.Compose([transforms.ToTensor()])

sampling=["ML_adapt","ML_lhs","ML_uniform"]
model_list=["2200.pt"]
layers=[2,2,2,2]
device='cuda' if torch.cuda.is_available() else 'cpu'

rootdir="./adaptive_sampling"
path2= rootdir + "/test/label/"
path1=rootdir + "/test/figures/"

tset=[3976, 206, 3669] # indices of best or worst results

label_list = []
pred_list = []
err_list = []

# ----------------------------------------------------------------------
# ----- How to read the saved model, predict data, and plot the results.
# ----------------------------------------------------------------------

for s in range(3):
    dummy = sampling[s] #sampling method
    for m in model_list:

        # Read the saved model
        PATH = rootdir + "/" + dummy + "/saved_models/" + m
        print(PATH)
        saved_model=M.ResNet(M.BasicBlock,layers).to(device)
        saved_model.load_state_dict(torch.load(PATH))
        #print(saved_model)

        tsy_pred_list = []
        tsy_label_list = []

        # Prediction
        for i in tset:
            # image
            fname = str(i) + ".png"
            img_path = os.path.join(path1, fname)

            img = Image.open(img_path).convert("L")  # greyscale
            img = trans(img)
            img.unsqueeze(0).shape
            pred = saved_model(img.unsqueeze(0).to(device))
            y_pred = pred.squeeze(0).detach().cpu().numpy()
            tsy_pred_list.append(y_pred)

            # labels
            fname2 = str(i) + ".txt"
            label_path = os.path.join(path2, fname2)

            data = []
            with open(label_path, "r") as f:
                for line in f:
                    a = np.float64(line.strip())
                    data.append(a)
                    TE = np.array(data)
                    label = TE
                y = label
                tsy_label_list.append(y)


        label_list.append(tsy_label_list)
        pred_list.append(tsy_pred_list)

        tmp=np.subtract(tsy_pred_list,tsy_label_list)
        loss_list=np.power(tmp,2)
        loss = loss_list.sum(axis=1)
        err_list.append(loss)


for i in range(3):
    plt.plot(err_list[i])
plt.show()

#a=lost_list.sum(axis=1)
#err_list

# ----------------------------------------------------------------------
# --- Plot the predicted transmission probabilities and simulation results.
# ----------------------------------------------------------------------
names=["Adaptive","LHS","MCS"]
colors=["red", "rebeccapurple", "green"]

x=[i for i in np.linspace(-1.25,1.25,1000)]  # convert the results in proper energy range

fig,axes=plt.subplots(1,3,figsize=(12,4))
for i in range(0,3): # three data
    axes[i].plot(x, pred_list[i][i], label=names[i], color=colors[i], linewidth=2)
    axes[i].plot(x, label_list[0][i], label="simulation", color="blue", linewidth=2, linestyle="dashed")

for i in range(3):
    axes[i].yaxis.set_tick_params(labelsize=16)
    axes[i].xaxis.set_tick_params(labelsize=16)
    axes[i].yaxis.set_tick_params(labelsize=16)
    axes[i].set_xlabel("E (eV)", fontsize=18)
    axes[i].set_ylabel("T(E)", fontsize=18)
    axes[i].legend(fontsize=12)
    #axes[i].set_ylim(0,1.0)
fig.tight_layout()
plt.show()
