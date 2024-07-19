import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.v2 import Lambda
import numpy as np
import os
import json

def unpack(X, lastdim=20):
    return np.unpackbits(X, axis=-1)[:,:,:lastdim] #, count=lastdim)

def remap_values(remapping, x):
    remapping=remapping[0].to(x.device),remapping[1].to(x.device)
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)

def get_ising_dataset(batch_size, train=None, train_size =500000, test_size=120000, dataset='Ising', drop_last=True, dataset_path=None):

    #Ts = np.arange(1.0,4.1,0.1)
    if dataset_path != None:
        with open(dataset_path + 'entropies.json','r') as json_file:
            ground_truth = json.load(json_file)
            Tss = [str(T) for T in ground_truth.keys()]   
    elif dataset=='gaussian':
        Ts=np.array([0, 1, 2, 3, 4])#0, 1, 2, 3, 4]
        Tss = [str(T) for T in Ts]
    elif dataset == 'xy':
        Ts = np.linspace(0.1, 1.5, 32)
        Tss = [str(round(T, 2)) for T in Ts]
    else:
        Ts = np.arange(1.0,4.1,0.1)
        Tss = ['{:.1f}'.format(T) for T in Ts]

      
    dataset_X = torch.Tensor()
    dataset_Y = torch.Tensor()
    dataset_size=train_size+test_size
    dataset_temperature_size = int(dataset_size/len(Tss))

    for T in Tss:
        #minde\ising_model_configurations\configurations_1.0_glauber.npz
        if dataset_path != None:
            Xb = np.load(dataset_path + 'configurations_{}.npy'.format(T))
            X_tensor = torch.Tensor(Xb)
        elif dataset=='Ising':
            Xb = np.load(os.getcwd() + '/entropy/my_model_configurations/configurations_{}.npy'.format(T))
            X_tensor = torch.Tensor(Xb)
            ground_truth = {1.0:0.007099426612215321,
                    1.1:0.01146732931882143,
                    1.2:0.01818824292310922,
                    1.3:0.027788734944455374,
                    1.4:0.04077254115784996,
                    1.5:0.05763654622830178,
                    1.6:0.07890241440418048,
                    1.7:0.10516662447951562,
                    1.8:0.1371807070022324,
                    1.9:0.17599393424000678,
                    2.0:0.2232529916578112,
                    2.1:0.28198699276557826,
                    2.2:0.3583482431774795,
                    2.3:0.45664861309821514,
                    2.4:0.5537154836509522,
                    2.5:0.6238005674883291,
                    2.6:0.672388132479363,
                    2.7:0.7090586955115314,
                    2.8:0.7385560991351459,
                    2.9:0.7631229859066431,
                    3.0:0.7840032070151927,
                    3.1:0.8019942333921092,
                    3.2:0.8176565292852906,
                    3.3:0.831406009233208,
                    3.4:0.8435619863869799,
                    3.5:0.854375387959015,
                    3.6:0.8640468687140126,
                    3.7:0.8727391323502708,
                    3.8:0.8805856514574879,
                    3.9:0.8876970182212816,
                    4.0:0.8941656727008463}
        elif dataset =='Ising_3x3':
            Xb = np.load(os.getcwd() + '/entropy/ising_3x3_configurations/configurations_{}.npy'.format(T))
            X_tensor = torch.Tensor(Xb)
            ground_truth = {1.0:0.007099426612215321,
                    1.1:0.01146732931882143,
                    1.2:0.01818824292310922,
                    1.3:0.027788734944455374,
                    1.4:0.04077254115784996,
                    1.5:0.05763654622830178,
                    1.6:0.07890241440418048,
                    1.7:0.10516662447951562,
                    1.8:0.1371807070022324,
                    1.9:0.17599393424000678,
                    2.0:0.2232529916578112,
                    2.1:0.28198699276557826,
                    2.2:0.3583482431774795,
                    2.3:0.45664861309821514,
                    2.4:0.5537154836509522,
                    2.5:0.6238005674883291,
                    2.6:0.672388132479363,
                    2.7:0.7090586955115314,
                    2.8:0.7385560991351459,
                    2.9:0.7631229859066431,
                    3.0:0.7840032070151927,
                    3.1:0.8019942333921092,
                    3.2:0.8176565292852906,
                    3.3:0.831406009233208,
                    3.4:0.8435619863869799,
                    3.5:0.854375387959015,
                    3.6:0.8640468687140126,
                    3.7:0.8727391323502708,
                    3.8:0.8805856514574879,
                    3.9:0.8876970182212816,
                    4.0:0.8941656727008463}
        elif dataset =='Spin':
            Xb = unpack(np.load(os.getcwd() + '/entropy/spin_glass_configurations/configurations_{}.npz'.format(T))['arr_0'])
            X_tensor[X_tensor==0]=-1.0
            X_tensor = torch.Tensor(Xb)
        elif dataset == 'xy':
            Xb = np.load(os.getcwd() + '/entropy/xy_model_configurations/configurations_{}.npz'.format(T))['arr_0']
            Xb = Xb/np.pi
            X_tensor = torch.Tensor(Xb)
            ground_truth = {0.1:-2.14075325683214,
                    0.15:-1.93124627775589,
                    0.19:-1.80738994121821,
                    0.24:-1.68323899253371,
                    0.28:-1.60013602230973,
                    0.33:-1.51011733656777,
                    0.37:-1.44629697756779,
                    0.42:-1.37429059736128,
                    0.46:-1.32157458776799,
                    0.51:-1.26035093912204,
                    0.55:-1.21443043240402,
                    0.6:-1.15996629429804,
                    0.64:-1.11816650013631,
                    0.69:-1.06742876458533,
                    0.73:-1.02755336695771,
                    0.78:-0.978017637648937,
                    0.82:-0.937980946519127,
                    0.87:-0.886477840965503,
                    0.91:-0.843120358534453,
                    0.96:-0.782909464400438,
                    1.0:-0.72841460798774,
                    1.05:-0.658120120082716,
                    1.09:-0.604554450354122,
                    1.14:-0.543478553426992,
                    1.18:-0.499842420227392,
                    1.23:-0.451489009352752,
                    1.27:-0.417424133910427,
                    1.32:-0.379655599089646,
                    1.36:-0.35290996007813,
                    1.41:-0.323216449283932,
                    1.45:-0.302060749329283,
                    1.5:-0.278413333905007}
        elif dataset == 'gaussian':
            Xb = np.load(os.getcwd() + '/entropy/gaussian_configurations/configurations_{}.npz'.format(T))['arr_0']
            X_tensor = torch.Tensor(Xb)
            #ground_truth = {0. : 3.178136190581754, 1. : 3.312397493796105, 2. : 3.195124867195717, 3. : 3.1085691402445264, 4. : 3.339988428272188}#0. : 3.178136190581754, 1. : 3.312397493796105, 2. : 3.195124867195717, 3. : 3.1085691402445264, 4. : 3.339988428272188}
            ground_truth = {0.: 4.626510225497235, 1.: 4.655866291475474, 2.: 4.6445175430284555, 3.: 4.620865461289007, 4.: 4.655177574457502}
            #ground_truth = {0.: 2.843763820697534, 1. : 3.063132189243023, 2. : 2.7101086999375585, 3. : 2.8453451011754436, 4. : 3.022376410130881}
    
        else:
            print('Unknown dataset')
        
        X_tensor = X_tensor[:dataset_temperature_size]
        
        T_tensor = float(T)*torch.ones(X_tensor.size(0))
        dataset_X = torch.cat([dataset_X, X_tensor])
        dataset_Y = torch.cat([dataset_Y, T_tensor])

    dataset_X = dataset_X.unsqueeze(1)
    dataset_Y = dataset_Y.unsqueeze(1)
    remapping = torch.unique(dataset_Y), torch.arange(len(torch.unique(dataset_Y)), dtype=torch.int64)
    dataset_Y = remap_values(remapping, dataset_Y)
    Ising_dataset = torch.utils.data.TensorDataset(dataset_X, dataset_Y)

    train_dataset, test_dataset = torch.utils.data.random_split(Ising_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=8, drop_last=drop_last, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False, pin_memory=True)
    
    return train_loader, test_loader, remapping, ground_truth
    