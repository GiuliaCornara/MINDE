import torch
import numpy as np
import math
import torchvision


def log_modalities(logger,output,mod_list,epoch,nb_samples=4, prefix="sampling/"):
    
    for mod in mod_list:
        if mod == "x":
            data_mod = output[mod].cpu()[:nb_samples,]
            dim=int(math.sqrt(torch.numel(data_mod)/nb_samples))
            ready_to_plot = torchvision.utils.make_grid( data_mod.view(data_mod.size(0), 1, dim, dim), nb_samples  )
            logger.experiment.add_image(prefix + mod, ready_to_plot, global_step=epoch)
        else:
            data_mod = output[mod].cpu()[:nb_samples,]
            temp_dict={}
            for i, el in enumerate(data_mod):
                temp_dict['T_{current_temperature}'.format(current_temperature=str(i))]=el[0]
            logger.experiment.add_scalars(prefix + mod, temp_dict, global_step=epoch)

def deconcat(z, mod_list, sizes):
    z_mods = {}
    idx = 0
    for i, mod in enumerate(mod_list):
        z_mods[mod] = z[:, idx:idx + sizes[i]]
        idx += sizes[i]
    return z_mods


def concat_vect(encodings):
    z = torch.Tensor()
    for key in encodings.keys():
        z = z.to(encodings[key].device)
        z = torch.cat([z, encodings[key]], dim=-1)
    return z


def unsequeeze_dict(data):
    for key in data.keys():
        if data[key].ndim == 1:
            data[key] = data[key].view(data[key].size(0), 1)
    return data
