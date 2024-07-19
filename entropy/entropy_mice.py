from tabnanny import verbose
import torch
#import gin
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import h5py
import numpy as np
import os
#import mice
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from tqdm import tqdm
import wandb
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
#import numba
from itertools import combinations_with_replacement
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--T',  type=float, default=0.1)
parser.add_argument('--resume_training', help='Boolean flag.', type=eval, choices=[True, False], default='True')

#@gin.configurable
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        return self.optimizer

#@gin.configurable
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss > self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class MiceConv(nn.Module):
    def __init__(self, input_size=576, kernel_size=3):
        super().__init__()
        #print(f'The input size of the fc will be {input_size}')
  
        self.kernel_size = kernel_size
        self.input_size = input_size

        self.layer1 =  nn.Sequential(nn.Conv2d(1, 8, kernel_size=self.kernel_size, stride=1, padding=0),
                                     nn.ReLU())


        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=self.kernel_size, stride=1, padding=0),
                                    nn.ReLU())
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop_out = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(input_size, max(1,int(input_size/2)))
        self.fc2 = nn.Linear(max(1, int(input_size/2)),  1)

        print('Finished init')

    def forward(self, data):

        output = self.layer1(data)

        if output.shape[-1]>self.kernel_size and output.shape[-2]>self.kernel_size:
          output = self.layer2(output)
        if output.shape[-1]>2 and output.shape[-2]>2:
          output = self.pooling(output)

        output = self.drop_out(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output

    

class MiceDataset(Dataset):
    '''
    Defining the Dataset to be used in the DataLoader

    x_joint: the lattices we got for the joint
    x_product: the lattices we got for the product
    n_samples: the number of lattices we got
    '''

    def __init__(self, x_joint, x_product):
        self.x_joint = x_joint
        self.x_product = x_product
        self.n_samples = x_joint.shape[0]

    def __getitem__(self, item):
        return self.x_joint[item], self.x_product[item]

    def __len__(self):
        return self.n_samples
    
def frames(T):
    '''
    Calculating the number of frames in the input

    return:
    the number of frames
    '''
    blocks = np.load(os.getcwd() + '/entropy/xy_model_configurations/configurations_{}.npz'.format(str(T)))['arr_0']
    num_frames = blocks.shape[0]  # number of frames in the input
    print(f'The number of frames in the input is: {num_frames}',
          f'\n',
          '='*50)
    return num_frames

def mi_model(genom, n_epochs, max_epochs, input_size=100, sizes=(10,10,10), kernel_size=3):
    '''
    Declare the model and loading the weights if necessary

    genom: the type of architecture we will use for the neural net
    n_epochs: number of epochs in the current run
    max_epochs: the maximum number of epochs we are using at the start, before we are using transfer learning

    return:
    the relevant model loaded with its weights
    '''
    # early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"
    weights_path = os.path.join('./', 'src', 'model_weights')

    if genom == 'mice_conv':
        model = MiceConv(input_size=input_size, kernel_size=kernel_size)#mice.MiceConv()
        model.to(device)

    if n_epochs != max_epochs and genom == 'mice_conv':
        print(f'==== mice_conv ====\nWeights have been loaded!\nWe are using {gpu_name}')
        PATH = os.path.join(weights_path, 'mice_conv''_'+str(sizes[0])+'_'+str(sizes[1])+'_'+'model_weights.pth')
        model = MiceConv(input_size=input_size, kernel_size=kernel_size)#mice.MiceConv()
        model.load_state_dict(torch.load(PATH), strict=False)
        model.eval()
        model.to(device)
    elif n_epochs == max_epochs and genom == 'mice_conv':
        PATH = os.path.join(weights_path, 'mice_conv_model_weights.pth')
        print(f'==== mice_conv ====\nThere are no weights, this is the first run!\nWe are using {gpu_name}')

    return model

#@gin.configurable
def boxes_maker( num_boxes, sample, T, flag=0):#num_samples, samples_per_snapshot
    '''
    Generating the sliced box of the sample mentioned in {sample}

    num_boxes: the number of boxes we split our space to
    sample: number of sample we chose randomly to take our data from
    flag: whether to use the data from data.h5 (flag=0), random data (flag=1), or the data that reproduces log2 mutual information (flag=2)

    return:
    our splitted space into the number of boxes we defined
    '''
    """if flag == 0:
        borders = np.linspace(0, 1, num_boxes+1, endpoint=True)
        blocks = read_data(T)
        boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))

        df_particles = pd.DataFrame(
            blocks[sample],
            columns=['X', 'Y', 'Z']
        )

        x_bin = borders.searchsorted(df_particles['X'].to_numpy())
        y_bin = borders.searchsorted(df_particles['Y'].to_numpy())
        z_bin = borders.searchsorted(df_particles['Z'].to_numpy())

        g = dict((*df_particles.groupby([x_bin, y_bin, z_bin]),))

        g_keys = list(g.keys())

        for cntr, cor in enumerate(g_keys):
            boxes_tensor[cor[0]-1, cor[1]-1, cor[2]-1] = 1"""
    try:
        return blocks[sample]/np.pi
    except:
      if flag == 0:
          blocks = np.load(os.getcwd() + '/entropy/xy_model_configurations/configurations_{}.npz'.format(str(T)))['arr_0']
          boxes_tensor=blocks[sample]/np.pi

      elif flag == 1:
          boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
          flag_torch = torch.randint_like(torch.tensor(boxes_tensor), low=0, high=2)
          boxes_tensor[flag_torch == 1] = 1

      elif flag == 2:
          boxes_tensor = np.zeros((num_boxes, num_boxes, num_boxes))
          i = np.random.randint(low=0, high=boxes_tensor.shape[0])
          j = np.random.randint(low=0, high=boxes_tensor.shape[1])
          k = np.random.randint(low=0, high=boxes_tensor.shape[2])
          boxes_tensor[i, j, k] = 1

    return boxes_tensor

#@gin.configurable
def lattices_generator(num_samples, samples_per_snapshot, R, num_frames, num_boxes, sizes, T, cntr=0, lattices=None):
    lattices = []
    cntr = 0
    x_size, y_size = sizes
    while cntr < num_samples:
        if cntr % (samples_per_snapshot) == 0:
            num_sample = R.randint(num_frames)
            my_tensor = boxes_maker(num_boxes=num_boxes, sample=num_sample, T=T)#mice.boxes_maker(num_boxes=num_boxes, sample=num_sample)  # returns a tensor
        leny_x = my_tensor.shape[0]
        leny_y = my_tensor.shape[1]
        x_steps = leny_x - x_size
        y_steps = leny_y - y_size
        if x_steps == 0:
            i = 0
        else:
            i = R.randint(0, x_steps+1)
        if y_steps == 0:
            j = 0
        else:
            j = R.randint(0, y_steps+1)

        lattices.append(np.expand_dims(my_tensor[i:i+x_size, j:j+y_size], axis=0))#, k:k + z_size], axis=0))
        cntr += 1
    return lattices

#@gin.configurable
"""def lattices_generator(num_samples, samples_per_snapshot, R, num_frames, num_boxes, sizes, T, cntr=0, lattices=None):
    '''
    Generate the lattices that will be used in our neural net

    num_samples: number of samples we will have in each epoch
    samples_per_snapshot: the number of samples to take from each snapshot
    R: np.random.RandomState
    num_frames: number of frames we have in the data, from it we will pick 1 frame randomly to take our data from
    num_boxes: the number of boxes we split our space to
    sizes: the sizes of the box we are calculating the mutual information to
    cntr: just a counter
    lattice: a list we will put the lattices we will construct into

    return:
    list of lattices we've constracted
    '''
    if lattices is None:
        lattices = []
    x_size, y_size = sizes
    num_sample = R.randint(num_frames)
    my_tensor = boxes_maker(num_boxes=num_boxes, sample=num_sample, T=T)#mice.boxes_maker(num_boxes=num_boxes, sample=num_sample)  # returns a tensor
    #print(my_tensor.shape)
    leny_x = my_tensor.shape[0]
    leny_y = my_tensor.shape[1]
    #leny_z = my_tensor.shape[2]
    x_steps = leny_x - x_size
    y_steps = leny_y - y_size
    #z_steps = leny_z - z_size
    while True:
        if x_steps == 0:
            i = 0
        else:
            i = R.randint(0, x_steps+1)
        if y_steps == 0:
            j = 0
        else:
            j = R.randint(0, y_steps+1)
        #if z_steps == 0:
        #    k = 0
        #else:
        #    k = R.randint(0, z_steps+1)

        lattices.append(np.expand_dims(my_tensor[i:i+x_size, j:j+y_size], axis=0))#, k:k + z_size], axis=0))
        cntr += 1
        #print(cntr)
        #print(np.array(lattices).shape)
        if cntr == num_samples:
            return lattices
        elif cntr % (samples_per_snapshot) == 0:
            #adding num_samples=50, samples_per_snapshot=10
            return lattices_generator(num_samples=num_samples, samples_per_snapshot=samples_per_snapshot, R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes, T=T, cntr=cntr, lattices=lattices)
"""
def lattice_splitter(lattices, axis):
    '''
    Here we are splitting the lattices given in {lattices} on the axis given in the {axis}

    lattices: list of the lattices we've constructed
    axis: the axis we will split our lattices on

    return:
    left lattices and right lattices
    '''

    left_lattices, right_lattices = [], []
    for lattice in lattices:
        left_lattice, right_lattice = np.split(lattice, 2, axis=axis)
        left_lattices.append(left_lattice)
        right_lattices.append(right_lattice)
    return np.array(left_lattices), np.array(right_lattices)

def loss_function(joint_output, product_output):
    """
    calculating the loss function

    joint_output: the joint lattices we've constructed
    product_output: the product lattices we've constructed

    return:
    mutual: the mutual information
    joint_output: the joint lattices we've constructed
    exp_product: the exponent of the product_output
    """
    exp_product = torch.exp(product_output)
    mutual = torch.mean(joint_output) - torch.log(torch.mean(exp_product))
    return mutual, joint_output, exp_product

def train_one_epoch(window_size, epoch, train_losses, model, data_loader, optimizer, ma_rate=0.01, ma_et=1.0):
    '''
    train one epoch

    model: the model we will train
    data_loader: the data_loader that keeps our data
    optimizer: optimizer
    ma_rate: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    model.train()
    total_loss = 0
    total_mutual = 0
    for batch_idx, data in enumerate(data_loader):
        loss, mutual, ma_et = train_one_step(window_size, epoch, train_losses, model, data, optimizer, ma_rate, ma_et)
        total_loss += loss
        total_mutual += mutual
    total_loss = total_loss / len(data_loader)
    total_mutual = total_mutual / len(data_loader)
    if epoch > window_size*2:
        total_mutual = float(loss_lin_ave(current_loss=total_mutual.cpu().detach().numpy(), data=train_losses, window_size=window_size))
        #mice.loss_lin_ave(current_loss=total_mutual.cpu().detach().numpy(), data=train_losses, window_size=window_size))
        total_mutual = torch.tensor(total_mutual, requires_grad=True)
    return total_loss, total_mutual

def train_one_step(window_size, epoch, train_losses, model, data, optimizer, ma_rate, ma_et):#, ma_et=1.0
    '''
    train one batch in the epoch

    model: the model we will train
    data_loader: the data_loader that keeps our data
    optimizer: optimizer
    ma_rate: used in order to calculate the loss function
    ma_et: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    x_joint, x_product = data
    optimizer.zero_grad()
    joint_output = model(x_joint.float())
    #print(f'joint output {joint_output}')
    product_output = model(x_product.float())
    try:
        mutual, joint_output, exp_product = loss_function(joint_output, product_output)
    except Exception as e:
        print(f'The error is:\n{e}')
        return 'problem'
    #print(f'ma_et {ma_et}')
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(exp_product)

    loss_train = my_criterion(joint_output, ma_et, exp_product)
    #HAVE MODIFIED HERE
    #loss_train = -mutual 
    
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    return loss_train, mutual, ma_et 

def my_criterion(joint_output, ma_et, exp_product):
    return -(torch.mean(joint_output) - (1 / ma_et.mean()).detach() * torch.mean(exp_product))

def valid_one_epoch(window_size, epoch, valid_losses, model, data_loader, ma_rate=0.01, ma_et=1.0):
    '''
    validation of one epoch

    model: the model we will train
    data_loader: the data_loader that keeps our data
    ma_rate: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    model.eval()
    total_loss = 0
    total_mutual = 0
    for batch_idx, data in enumerate(data_loader):
        with torch.no_grad():
            loss, mutual, ma_et = valid_one_step(window_size, epoch, valid_losses, model, data, ma_rate, ma_et)
            total_loss += loss
            total_mutual += mutual
    total_loss = total_loss / len(data_loader)
    total_mutual = total_mutual / len(data_loader)
    if epoch > window_size*2:
        total_mutual = float(loss_lin_ave(current_loss=total_mutual.cpu().detach().numpy(), data=valid_losses, window_size=window_size))
        #mice.loss_lin_ave(current_loss=total_mutual.cpu().detach().numpy(), data=valid_losses, window_size=window_size))
        total_mutual = torch.tensor(total_mutual, requires_grad=True)
    return total_loss, total_mutual

def valid_one_step(window_size, epoch, valid_losses, model, data, ma_rate, ma_et):
    '''
    validation of one batch in the epoch

    model: the model we will train
    data_loader: the data_loader that keeps our data
    ma_rate: used in order to calculate the loss function
    ma_et: used in order to calculate the loss function

    return:
    loss and mutual information
    '''
    x_joint, x_product = data
    joint_output = model(x_joint.float())
    product_output = model(x_product.float())
    try:
        mutual, joint_output, exp_product = loss_function(joint_output, product_output)
    except Exception as e:
        print(f'The error is:\n{e}')
        return 'problem'
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(exp_product)
    loss_train = -(torch.mean(joint_output) - (1 / ma_et.mean()).detach() * torch.mean(exp_product))
    if epoch > window_size*2:
        loss_train = float(loss_lin_ave(current_loss=loss_train.cpu().detach().numpy(), data=valid_losses, window_size=window_size))
        #mice.loss_lin_ave(current_loss=loss_train.cpu().detach().numpy(), data=valid_losses, window_size=window_size))
        loss_train = torch.tensor(loss_train, requires_grad=True)
    return loss_train, mutual, ma_et

#@gin.configurable
def entropy_fig_running(x_labels, mi_entropy_dependant, mi_entropy_dependant_valid, genom, figsize):
    '''
    Plot the results together while the simulation is running

    x_labels: the sizes of the boxes of the calculation
    mi_entropy_dependant: the mutual informations together of the training
    mi_entropy_dependant_valid: the mutual informations together of the validation
    genom: the type of architecture we've trained our neural net with
    figsize: the size of the figure

    return:
    None
    '''
    plt.figure(num=len(x_labels)+2, figsize=figsize)
    plt.clf()
    plt.xlabel('size of the small box')
    plt.ylabel('Mutual Information')
    plt.title('Entropy searching...')
    plt.tight_layout()
    plt.plot(x_labels, mi_entropy_dependant, label=(genom + ' - train'))
    plt.plot(x_labels, mi_entropy_dependant_valid, label=(genom + ' - valid'))
    plt.legend()
    saved_path = os.path.join('./', "figures", "losses", "entropy_calculation")
    folder_checker(saved_path)#mice.folder_checker(saved_path)
    plt.savefig(fname=os.path.join(saved_path, 'simulation_running'))

    #@gin.configurable
def logger(my_str, mod, flag=[], number_combinations=0, flag_message=0, num_boxes=0):
    '''
    prints the results

    my_str: the string to be plotted
    mod: mod 0 prints both | mod 1 : prints only output | mod 2 : prints only to file
    flag: if it is first time printing
    number_combinations: the lengh of our printing
    flag_message: if we are printing the box size searching or the entropy calculation
    '''
    message_path = os.path.join('./', 'src', 'mice')
    folder_checker(message_path)#mice.folder_checker(message_path)
    if flag_message == 0:
        message_path = os.path.join(message_path, 'message_boxcalc.log')
    elif flag_message == 1:
        message_path = os.path.join(message_path, 'message_entropycalc.log')
    elif flag_message == 2:
        message_path = os.path.join(message_path, 'message_isingcalc.log')

    try:
        logger.counter += 1
    except Exception:
        logger.counter = 0

    if flag == []:
        flag.append('stop')
        log_file = open(message_path, "w")
        sys.stdout = log_file
        if flag_message == 0:
            print(f'==== log file for the Mutual Information for different number of boxes ====\n\n'
                  f'We have {number_combinations} runs in total\n\n')
        elif flag_message == 1:
            print(f'==== log file for the Mutual Information for different box shapes ====\n\n'
                  f'We split our space into: {num_boxes} boxes.\nWe have {number_combinations} runs in total\n\n')
        elif flag_message == 2:
            print(f'==== log file for the Mutual Information for ising ====\n\n')
        sys.stdout = sys.__stdout__
        log_file.close()

    if mod == 0:
        log_file = open(message_path, "a+")
        sys.stdout = log_file
        print(logger.counter,". ",my_str,"\n")
        sys.stdout = sys.__stdout__
        log_file.close()
        print(my_str)
    elif mod == 1:
        print(my_str)
    elif mod == 2:
        log_file = open(message_path, "a+")
        sys.stdout = log_file
        print(logger.counter,". ",my_str,"\n")
        sys.stdout = sys.__stdout__
        log_file.close()

def folder_checker(path):
    '''
    if a folder in {path} does not exist, this function will create it

    return:
    None
    '''
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    return None

def sort_func(args):
    '''
    calculating the i*j*k of the input

    return:
    i*j*k
    '''
    i,j,k = args
    return i*j*k

def print_combinations(my_combinations):
    '''
    printing all of the combinations

    my_combinations: or combinations

    return:
    None
    '''
    print(f'All of my combinations are:')
    for cntr, i in enumerate(my_combinations):
        print(f'{cntr}. {i}')
    return None

#@gin.configurable
def exp_ave(data, window_frac):
    data = np.array(data)
    window = np.floor(data.shape[0] * window_frac).astype(int)
    ave_arr = np.zeros((data.shape[0]))
    mi = data[0]
    alpha = 2 / float(window + 1)
    for i in range(data.shape[0]):
        mi =  ((data[i] - mi) * alpha) + mi
        ave_arr[i] = mi
    return ave_arr

#@gin.configurable
def lin_ave(data, window_frac):
    data = np.array(data)
    window = np.floor(data.shape[0] * window_frac).astype(int)
    return [np.mean(data[i:i+window]) for i in range(0,len(data)-window)]

def lin_ave_running(epoch, data, window_size):
    data = np.array(data)
    if epoch == 0:
        return [0]
    elif epoch < window_size:
        return [np.mean(data[i:i+epoch]) for i in range(0,len(data)-epoch)]
    return [np.mean(data[i:i+window_size]) for i in range(0,len(data)-window_size)]

def loss_lin_ave(current_loss, data, window_size):
    current_data = data.copy()
    current_data.append(current_loss)
    current_data = np.array(current_data)
    return current_data[-window_size:].mean()

def loss_exp_ave(current_loss, data, window_size):
    weights = np.linspace(1, 10, 10, dtype='int')
    weights = np.exp(weights)
    current_data = data.copy()
    current_data.append(current_loss)
    current_data = np.array(current_data)
    return np.ma.average(current_data, weights=weights)


#@gin.configurable
def entropy_runner(num_boxes, idx, comb, number_combinations, max_epochs, batch_size, freq_print, genom, lr, weight_decay, num_samples, transfer_epochs, T, resume_training, window_size=3):
    '''
    Running the neural network in order to calculate the right number of boxes to split our space into

    box_sizes: the number of boxes in each axis to split our space into
    max_epochs: the maximum number of epochs to use in the beginning
    batch_size: the size of the batch
    freq_print: the number of epochs between printing to the user the mutual information
    axis: the axis we will split our boxes into, in order to calculate the mutual information
    genom: the type of architecture we are going to use in the neural net
    lr: the learning rate
    weight_decay: regularization technique by adding a small penalty
    box_frac: what is the value of the box from the total space we are calculating the mutual information to

    return:
    None
    '''

    weights_path = os.path.join('./', 'src', 'model_weights')
    PATH = os.path.join(weights_path, genom+'_'+str(comb[0])+'_'+str(comb[1])+'_model_weights.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R = np.random.RandomState(seed=0)

    # num_frames = mice.frames()
    cntr = 0
    mi_entropy_dependant = []
    mi_entropy_dependant_valid = []

    x_labels = []
    saved_directory = os.path.join('./data', f'{num_boxes}')

    # for idx, (i, j, k) in enumerate(my_combinations):
    if resume_training == False: n_epochs = max_epochs
    else: n_epochs = transfer_epochs

    wandb.login()
    config = dict(
    epochs=n_epochs,
    batch_size=batch_size,
    learning_rate=lr,
    weight_decay=weight_decay,
    dataset="my_data",
    architecture=genom,
    combination=comb,
    temperature=T)

    i, j = comb
    sizes = (i, j)
    kernel_size = 3
    if i<kernel_size or j<kernel_size:
      kernel_size=min(i,j)
    print(sizes)
    #with h5py.File(os.path.join(saved_directory, f'{i}_{j}_{k}', 'data.h5'), "r") as hf:
        #lattices = np.array(hf.get('dataset_1'))
    # lattices = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
    #lattices = np.load('/content/configurations_0.1.npz')['arr_0']
    #lattices = np.expand_dims(lattices, 1)
    num_frames = frames(T)
    if i<8 and j<8:
        num_samples_per_snapshot=10
    elif i<16 and j <16:
        num_samples_per_snapshot=4
    else:
        num_samples_per_snapshot=2
    

    #Small num_samples otherwise all RAM in colab is used
    lattices=lattices_generator(num_samples=num_samples, samples_per_snapshot=num_samples_per_snapshot, R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes, T=T)
    lattices=np.array(lattices)
    
    print(lattices.shape)

    list_ising = lattices.copy()
    print(list_ising.shape)
    x_size = list_ising[0].shape[1]
    y_size = list_ising[0].shape[2]
    #z_size = list_ising[0].shape[3]
    # input_size = x_size * y_size * z_size
    #input_size = int(8 * ((x_size-2)/1+1) * ((y_size-2)/1+1))# * ((z_size-2)/1+1)
    #input_size = int(4 * (x_size-4) * (y_size-4))
    
    #After first convolution
    size_x = (x_size-kernel_size+1)
    size_y=(y_size-kernel_size+1)
    mult_channels = 8
    if size_x>kernel_size and size_y>kernel_size:
      #After second convolution
      size_x = (size_x - kernel_size+1)
      size_y = (size_y - kernel_size+1)
      mult_channels = 16
  
      if size_x>2 and size_y>2:
          size_x = size_x/2
          size_y = size_y/2
    
    input_size = int(mult_channels*size_x*size_y)#int(4 * (size_x-4) * (size_y-4))
    
    print(f'Computed input size will be {input_size}')

    axis = int(np.argmax((i, j)))
    print('='*50)
    print(f'The size of the small boxes is: {i}x{j}\n'
            f'Therefore we cut on the {axis} axis\n'
            f'Building the boxes... we are going to start training...')
    print(len(list_ising))
    print(num_samples)
    axis += 1

    model = mi_model(genom=genom, n_epochs=n_epochs, max_epochs=max_epochs, input_size=input_size, kernel_size=kernel_size, sizes=sizes)#mice.mi_model(genom=genom, n_epochs=n_epochs, max_epochs=max_epochs, input_size=input_size)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = LRScheduler(optimizer)#mice.LRScheduler(optimizer)
    early_stopping = EarlyStopping()#mice.EarlyStopping()
    train_losses = []
    valid_losses = []
    with wandb.init(project="mice project", config=config):
        wandb.watch(model, my_criterion, log="all", log_freq=1000)#mice.my_criterion, log="all", log_freq=1000)
        print(f'Temperature: {T}')
        print(f'Combination: {comb}')
        for epoch in tqdm(range(int(n_epochs))):
            # lattices = mice.lattices_generator(R=R, num_frames=num_frames, num_boxes=num_boxes, sizes=sizes)
            # list_ising = lattices.copy()
            place = random.sample(range(len(list_ising)), k=int(num_samples))
            lattices = np.array(list_ising)[place]
            left_lattices, right_lattices = lattice_splitter(lattices=lattices, axis=axis)#mice.lattice_splitter(lattices=lattices, axis=axis)
            joint_lattices = np.concatenate((left_lattices, right_lattices), axis=axis + 1)
            right_lattices_random = right_lattices.copy()
            R.shuffle(right_lattices_random)
            product_lattices = np.concatenate((left_lattices, right_lattices_random), axis=axis + 1)
            joint_lattices, joint_valid, product_lattices, product_valid = train_test_split(joint_lattices, product_lattices,
                                                                                            test_size=0.2, random_state=42)
            AB_joint, AB_product = torch.tensor(joint_lattices), torch.tensor(product_lattices)
            AB_joint_train, AB_product_train = AB_joint.to(device), AB_product.to(device)
            dataset_train = MiceDataset(x_joint=AB_joint_train, x_product=AB_product_train)#mice.MiceDataset(x_joint=AB_joint_train, x_product=AB_product_train)

            AB_joint, AB_product = torch.tensor(joint_valid), torch.tensor(product_valid)
            AB_joint_valid, AB_product_valid = AB_joint.to(device), AB_product.to(device)
            dataset_valid = MiceDataset(x_joint=AB_joint_valid, x_product=AB_product_valid)#mice.MiceDataset(x_joint=AB_joint_valid, x_product=AB_product_valid)

            loader = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=0, shuffle=False)
            loss_train, mutual_train = train_one_epoch(window_size=window_size, epoch=epoch, train_losses=train_losses, model=model, data_loader=loader, optimizer=optimizer)
           #mice.train_one_epoch(window_size=window_size, epoch=epoch, train_losses=train_losses, model=model, data_loader=loader, optimizer=optimizer)
            train_losses.append(mutual_train.cpu().detach().numpy())

            loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, num_workers=0, shuffle=False)
            valid_loss, valid_mutual = valid_one_epoch(window_size=window_size, epoch=epoch, valid_losses=valid_losses, model=model, data_loader=loader)
            #mice.valid_one_epoch(window_size=window_size, epoch=epoch, valid_losses=valid_losses, model=model, data_loader=loader)
            valid_losses.append(valid_mutual.cpu().detach().numpy())

            train_losses_exp = list(exp_ave(data=train_losses, window_frac=0.1))#mice.exp_ave(data=train_losses))
            valid_losses_exp = list(exp_ave(data=valid_losses, window_frac=0.1))#mice.exp_ave(data=valid_losses))
            train_losses_exp = list(exp_ave(data=train_losses_exp, window_frac=0.1))#mice.exp_ave(data=train_losses_exp))
            valid_losses_exp = list(exp_ave(data=valid_losses_exp, window_frac=0.1))#mice.exp_ave(data=valid_losses_exp))

            if epoch > 500:
                lr = lr_scheduler(train_losses_exp[-1]).param_groups[0]["lr"]
                early_stopping(train_losses_exp[-1])
                if early_stopping.early_stop:
                    break
                elif epoch > 2000 and train_losses_exp[-1] < 1e-6:
                    break

            if epoch % freq_print == 0:
                print(f'\nMI for train {train_losses_exp[-1]}, val {valid_losses_exp[-1]} at step {epoch}')
                wandb.log({"epoch": epoch, "train loss": train_losses_exp[-1], "valid loss": valid_losses_exp[-1],
                "learning rate":lr, "batch size": batch_size})


        cntr += 1
        x_labels.append(str((i, j)))
        folder_checker(weights_path)#mice.folder_checker(PATH)
        torch.save(model.state_dict(), PATH)
        # dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
        # torch.onnx.export(model, dummy_input, "model.onnx")
        wandb.save("model.onnx")
        train_losses =exp_ave(data=train_losses, window_frac=0.1) #mice.exp_ave(data=train_losses)
        valid_losses = exp_ave(data=valid_losses, window_frac=0.1)#mice.exp_ave(data=valid_losses)
        train_losses = exp_ave(data=train_losses, window_frac=0.1)#mice.exp_ave(data=train_losses)
        valid_losses = exp_ave(data=valid_losses, window_frac=0.1)#mice.exp_ave(data=valid_losses)
        mi_entropy_dependant.append(train_losses[-1])
        mi_entropy_dependant_valid.append(valid_losses[-1])
        # mice.entropy_fig(num=cntr, genom=genom, sizes=sizes, train_losses=train_losses, valid_losses=valid_losses)
        logger(f'The MI train for ({i}, {j}) box is: {train_losses[-1]:.2f}', number_combinations=number_combinations, flag_message=1, mod=0, num_boxes=num_boxes)
        print('logged')
        #mice.logger(f'The MI train for ({i}, {j}, {k}) box is: {train_losses[-1]:.2f}', number_combinations=number_combinations, flag_message=1, num_boxes=num_boxes)
        print('take a look:')
        print(train_losses[-1], valid_losses[-1], genom)
        return train_losses[-1], valid_losses[-1], genom

if __name__ == '__main__':
    args = parser.parse_args()
    num_boxes = 20
    limit = 5096
    T=args.T
    resume_training = args.resume_training
    

    my_root = int(np.floor(np.log2(num_boxes)))
    temporal_combinations = list(combinations_with_replacement([2 << expo for expo in range(0, my_root)], 2))

    print('Our combinations are:')
    # temporal_combinations = [i for i in my_combinations if math.prod(i) < limit]
    my_combinations = list()

    for i in temporal_combinations:
        #if (i[0] >= 4 and i [1] >= 4):# and (i[0] <= 16 and i[1] <= 16):
    #if (i[0] == 15 and i [1] == 15 and i [2] == 15):
        if (i[0]==2*i[1] or i[1]==2*i[0] or i[0]==i[1]):
            my_combinations.append(i)
    my_combinations.append((2,1))
    my_combinations.sort(key=lambda x: math.prod(x))
    print_combinations(my_combinations)#mice.print_combinations(my_combinations)
    number_combinations = len(my_combinations)
    x_labels = []
    mi_entropy_dependant = []
    mi_entropy_dependant_valid = []
    mi_divided_by_area=[]
    mi_divided_by_area_valid=[]
    for idx, (i, j) in enumerate(my_combinations):
        x_labels.append(str((i, j)))
        comb = (i, j)
        train_loss, valid_loss, genom = entropy_runner(num_boxes=num_boxes, idx=idx, comb=comb, number_combinations=number_combinations, resume_training=resume_training, max_epochs=2500, freq_print=100, genom = 'mice_conv', weight_decay=0, num_samples=2000, transfer_epochs=1000, T=T, window_size=3, lr = 1e-3, batch_size = 32)
        #mice.entropy_runner(num_boxes=num_boxes, idx=idx, comb=comb, number_combinations=number_combinations)
        mi_entropy_dependant.append(train_loss)
        mi_entropy_dependant_valid.append(valid_loss)
        mi_divided_by_area.append(train_loss/(i*j))
        mi_divided_by_area_valid.append(valid_loss/(i*j))#mice.entropy_fig_running(x_labels=x_labels, mi_entropy_dependant=mi_entropy_dependant, mi_entropy_dependant_valid=mi_entropy_dependant_valid, genom=genom)
        #mice.entropy_fig_running(x_labels=x_labels, mi_entropy_dependant=mi_entropy_dependant, mi_entropy_dependant_valid=mi_entropy_dependant_valid, genom=genom)
    # mice.entropy_fig_together(x_labels=x_labels, mi_entropy_dependant=mi_entropy_dependant, mi_entropy_dependant_valid=mi_entropy_dependant_valid, genom=genom)
    # mi_entropy_dependant = np.array(mi_entropy_dependant)
    # mice.logger(f'The total MI train is: {mi_entropy_dependant.sum():.2f}', number_combinations=number_combinations, flag_message=1)
    entropy_fig_running(x_labels=x_labels, mi_entropy_dependant=mi_entropy_dependant, mi_entropy_dependant_valid=mi_entropy_dependant_valid, genom=genom, figsize=(5,5))
    sum_entropy = np.array(mi_divided_by_area).sum()#The division per 2 is already taken into account by the size being i*j 
                                                    #while it should be i*j/2 -0.5*np.array(mi_divided_by_area).sum()
    print(f'The total entropy train is: {sum_entropy}')
    logger(f'The total entropy train is: {sum_entropy}', number_combinations=number_combinations, flag_message=1, mod=0)
    sum_entropy_valid =np.array(mi_divided_by_area_valid).sum()  
    print(f'The total entropy validation is: {sum_entropy_valid}')
    logger(f'The total entropy validation is: {sum_entropy_valid}', number_combinations=number_combinations, flag_message=1, mod=0)