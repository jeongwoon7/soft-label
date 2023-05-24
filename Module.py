""" --------------------------------------------------------------------------------------------------------
All the functions used in the machine learning process are defined here.
- Basically, the ResNet architecture implemented in PyTorch is used.
------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import glob
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchsummary import summary


"""[Custom dataset for data loading]------------------------------------------------------------------------
    Each data in train, valid, test sets consists of figure (*.png) and label (*.txt).
    A figure is a simplified 224x224 image of Quantum dot chain.
    A label is a transmission probability (as a function of energy) expressed by 1000 numerical data points.
------------------------------------------------------------------------------------------------------------
"""
class CustomDataset(Dataset):

    def __init__(self, path, train=False, valid=False, test=False, transform=transforms.ToTensor()):
        self.path = path
        flag = {"train": train, "valid": valid, "test": test}

        for key, value in flag.items():
            if value:
                self.img_path = path + "/figures/" + key
                self.label_path = path + "/label/" + key

        self.transform = transform
        # sort image and label files according to their names for input pairs
        self.img_list = sorted(glob.glob(self.img_path + '/*.png'))
        self.label_list = sorted(glob.glob(self.label_path + '/*.txt'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]

        img = Image.open(img_path).convert("L")  # greyscale
        image = self.transform(img)

        data = []
        with open(label_path, "r") as f:
            for line in f:
                a = np.float64(line.strip())
                data.append(a)
                TE = np.array(data)
        label = TE  # Transmission probability T(E) represented by 1000 data points

        return image, label


"""[ResNet class]-----------------------------------------------------------------------------------------
    Originally, ResNet was for classifying data into 1000 classes, i.e., num_classes=1000.
    Here, the 1000 numerical values are used for regression to predict transmission probability (label).
    Gray scale is used instead of RGB. So, the number of input channel to nn.Conv2d is 1 instead of 3.
-----------------------------------------------------------------------------------------------------------
"""

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    r"""
    3x3 convolution with padding
    - in_planes: in_channels
    - out_channels: out_channels
    - bias=False: bias=False
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        r"""
         - inplanes: input channel size
         - planes: output channel size
         - groups, base_width: ResNext or Wide ResNet
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Basic Block
        self.conv1 = conv3x3(inplanes, planes, stride)  # downsample
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # short connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # ResNext or WideResNet
        width = int(planes * (base_width / 64.)) * groups

        # Bottleneck Block
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # conv2에서 downsample
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 1x1 convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3 convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1 convolution layer
        out = self.conv3(out)
        out = self.bn3(out)
        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


"""-----------------------------------------------------------------------------------------------------
(1) Basic model shown in Fig.1 of the manuscript
- num_classes can be any numbers corresponding to the dimension of input label (e.g., 2000 data points)
--------------------------------------------------------------------------------------------------------
"""
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # default values
        self.inplanes = 64  # input feature map
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # the number of filters is doubled
        # the feature map size is halved
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    
""" -----------------------------------------------------------------------------------------------------
(2) Loss prediction model
- Now that the Loss prediction model predicts the loss of the basic model's prediction (a numerical value),
the num_classes is now 1.
---------------------------------------------------------------------------------------------------------
"""

class loss_pred_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(loss_pred_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # default values
        self.inplanes = 64  # input feature map
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,  # 여기서부터 downsampling적용
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # the number of filters is doubled: self.inplanes와 planes 사이즈를 맞춰주기 위한 projection shortcut
        # the feature map size is halved: stride=2로 downsampling
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion  # inplanes 업데이트

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


"""--------------------------------------------------------------------
# Train and valid loops for model training
- Predict, calculate errors, back-propagate
-----------------------------------------------------------------------
-----------------------------------------------------------------------
(1) Training loop for the basic model
- X is the tensor representing input 224x224 image
- Y is the tensor representing the input label (1000 data points)
-----------------------------------------------------------------------
"""

def train_loop(dataloader, model, loss_fn, optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0

    for batch, (x, y) in enumerate(dataloader):
  
        """ Compute prediction and loss
        # X.shape --> torch.Size([batch_size,1,224,224])
        # Y.shape --> torch.Size([batch_size,1000])
        """
        X = torch.as_tensor(x, dtype=torch.float, device=device)
        Y = torch.as_tensor(y, dtype=torch.float, device=device)

        pred = model(X)
        loss = 1000 * loss_fn(pred, Y)  # magnified the MSE error by 1000
        train_loss += loss.detach()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Avg loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches #size
    print(f"Train Error: Avg loss: {train_loss:>8f} \n")
    return train_loss


"""--------------------------------------------------------------------
(2) Training loop for the loss prediction model
-----------------------------------------------------------------------
"""
def train_loop2(dataloader, model, loss_fn, optimizer, model2, loss_fn2, optimizer2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    train_loss2=0

    for batch, (x, y) in enumerate(dataloader):
        
        """ Compute prediction and loss
        # X.shape --> torch.Size([batch_size,1,224,224])
        # Y.shape --> torch.Size([batch_size,1000])
        """
        X = torch.as_tensor(x, dtype=torch.float, device=device)
        Y = torch.as_tensor(y, dtype=torch.float, device=device)

        pred = model(X)
        loss = 1000*loss_fn(pred, Y)  # magnify the MSE error by 1000
        train_loss += loss.detach()

        #----------09/25/2022 for loss prediction-----------
        tmp=np.subtract(pred.detach().cpu(),Y.cpu()) # point-wise subtraction
        tmp=np.power(tmp,2)/len(tmp[-1])             
        error=[sum(tmp[i]) for i in range(len(tmp))]
        Y2 = torch.tensor(error, device=device).unsqueeze(1)  # label of the loss prediction model

        pred2=model2(X)             # prediction of the loss prediction model 
        loss2= loss_fn2(pred2,Y2)   # batch averaged loss
        train_loss2 += loss2.detach()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Backpropagation2
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        if batch % 100 == 0: 
            loss, current = loss.item(), batch * len(X)
            print(f"Avg loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches  # size
    train_loss2 /= num_batches
    print(f"Train Error: Avg loss: {train_loss:>8f}")           # error of the basic model
    print(f"Train Error2: Avg loss: {train_loss2:>10f} \n")     # error of the loss prediction model
    return train_loss, train_loss2


"""--------------------------------------------------------------------
(3) The common valid loop
-----------------------------------------------------------------------
"""
def valid_loop(dataloader, model, loss_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            X = torch.as_tensor(x, dtype=torch.float, device=device)
            Y = torch.as_tensor(y, dtype=torch.float, device=device)

            pred = model(X)
            valid_loss += 1000*loss_fn(pred, Y).item()

    valid_loss /= num_batches #size
    print(f"Valid Error: Avg loss: {valid_loss:>8f} \n")

    return valid_loss


