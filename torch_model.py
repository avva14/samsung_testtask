import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torchsummary import summary
from pytorch_msssim import ms_ssim

from thop import profile

class FramesDataset(Dataset):
    def __init__(self, path):
        #
        self.shortpath = os.path.join(path, 'short')
        self.medpath = os.path.join(path, 'medium')
        self.filelist = np.array([f for f in os.listdir(self.shortpath) if f.split('.')[-1] == 'bin'])
        self.filenumber = self.filelist.shape[0]
        return

    def __getitem__(self, index):
        filename = self.filelist[index]
        shortname = os.path.join(self.shortpath, filename)
        medname = os.path.join(self.medpath, filename)
        shortdata = np.fromfile(shortname, dtype = np.uint8).reshape((256,256,3))
        meddata = np.fromfile(medname, dtype = np.uint8).reshape((256,256,3))
        return shortdata, meddata

    def __len__(self):
        return self.filenumber

class BrightnessCorrector(nn.Module):
    def __init__(self, numll, numhl):

        super(BrightnessCorrector, self).__init__()

        self.nll = numll
        self.nhl = numhl

        self.firstconvrelu = nn.Conv2d(in_channels=3, out_channels=61, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode='reflect')
        self.firstconvth = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode='reflect')

        self.convrelu = nn.Conv2d(in_channels=64, out_channels=61, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode='reflect')
        self.convth = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3,3), stride=(1,1), padding=(1,1), padding_mode='reflect')

        self.convhl = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1), padding_mode='reflect')
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

        self.globpool = nn.AvgPool2d(16)
        self.flat = nn.Flatten()
        self.dns = nn.Linear(in_features=64, out_features=30)
        self.constadd = nn.ConstantPad2d((0,0,1,0), 1.0)
        self.triange = torch.tensor([
            [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            [[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            [[0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            [[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
            [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],
            [[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]],
            [[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]],
            [[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]],
            [[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]],
            [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]
            ], dtype=torch.float32)

        return

    def forward(self, input):

        x1 = F.relu(self.firstconvrelu(input))
        x2 = torch.tanh(self.firstconvth(input))
        img = input + x2
        out = torch.cat((x1, img), 1)

        for _ in range(self.nll-1):
            x1 = F.relu(self.convrelu(out))
            x2 = torch.tanh(self.convth(out))
            img = img + x2
            out = torch.cat((x1, img), 1)

        for _ in range(self.nhl):
            out = F.relu(self.convhl(out))
            out = self.pool(out)

        pooled = self.flat(self.globpool(out))
        densed = self.dns(pooled)

        img = torch.reshape(img, shape=[-1,3,256*256])
        img = self.constadd(img)

        rows = torch.reshape(densed, shape=[-1,3,10])
        tri = torch.einsum('...i,ijk->...jk', rows, self.triange)
        multimg = torch.einsum('aik,abij,ajk->abk', img, tri, img)
        res = torch.reshape(multimg, shape=[-1,3,256,256])

        return res

def safepowneg(x):
    return torch.pow(torch.abs(x),1/3)
def safepowpos(x):
    return torch.pow(torch.abs(x),2.4)

def labconvert(rgb):
    '''
    RGB to LAB
    '''
    matrix = torch.tensor([[0.43388193, 0.37622739, 0.18990225],
       [0.2126    , 0.7152    , 0.0722    ],
       [0.01772529, 0.1094743 , 0.87294736]], dtype=torch.float32)
    shmatrix = torch.tensor([[0, 116, 0], [500, -500, 0], [0, 200, -200]], dtype=torch.float32)
    val1 = torch.tensor(0.04045, dtype=torch.float32)
    val2 = torch.tensor(0.008856451679035631, dtype=torch.float32)
    p = torch.clip(rgb, min=0, max=1)
    f = torch.where(p <= val1, 0.07739938080495357 * p, safepowpos(0.9478672985781991 * (p + 0.055)))
    x = torch.einsum('ij,...j->...i', matrix, f)
    die = torch.where(x > val2, safepowneg(x), 0.13793103448275862 + 7.787037037037036 * x)
    fie = torch.einsum('ij,...j->...i', shmatrix, die)

    return fie

def labconvertL(rgb):
    '''
    RGB to LAB, withou AB
    '''
    matrix = torch.tensor([[0.2126    , 0.7152    , 0.0722    ]], dtype=torch.float32)
    val1 = torch.tensor(0.04045, dtype=torch.float32)
    val2 = torch.tensor(0.008856451679035631, dtype=torch.float32)
    
    p = torch.clip_by_value(rgb, clip_value_min=0, clip_value_max=1)
    f = torch.where(p <= val1, 0.07739938080495357 * p, safepowpos(0.9478672985781991 * (p + 0.055)))
    x = torch.einsum('ij,...j->...i', matrix, f)
    die = torch.where(x > val2, safepowneg(x), 0.13793103448275862 + 7.787037037037036 * x)
    fie = 116 * die - 16
    return fie

def MessyLoss(output, target):
    lout = labconvert(output)
    ltarget = labconvert(target)
    maeloss = torch.mean(torch.abs(lout - ltarget))
    lout1 = labconvertL(output)
    ltarget1 = labconvertL (target)
    ssimloss = 1-ms_ssim(lout1,ltarget1,100)
    return maeloss + ssimloss

def main():

    device = torch.device('cpu')

    model = BrightnessCorrector(numll=4, numhl=2).to(device=device)
    summary(model, (3, 256, 256))
    x = torch.zeros((4,3,256,256))
    y = model.forward(x)
    print(f'Output shape {y.shape}')

    macs, params = profile(model, inputs=(x, ))
    print(f'{macs /1000 /1000 /1000} GFLPS')

    return

    #data loader

    batch_size = 4
    frms_train = FramesDataset('train')
    train_loader = DataLoader(dataset=frms_train, batch_size=batch_size, shuffle=True)

     # training loop

    num_epochs = 2
    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    for epoch in range(num_epochs):
        for shortex, medex in train_loader:  
            shortex = shortex.to(device)
            medex = medex.to(device)
            
            # Forward pass
            outputs = model(shortex)
            loss = MessyLoss(outputs, medex)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__=='__main__':
    main()