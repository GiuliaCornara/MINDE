import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.model.score_net import UnetMLP
from src.model.transformer import DiT
from src.libs.ema import EMA
from src.libs.SDE import VP_SDE, concat_vect, deconcat
from src.libs.importance import get_normalizing_constant
from src.libs.util import log_modalities
from entropy.ising_data import ising_dataset
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import json
import argparse
import math
import numpy as np
from PIL import Image
import torchvision
import io
import time

T0 = 1


parser = argparse.ArgumentParser()
parser.add_argument('--rows',  type=int, default=0)
parser.add_argument('--seed',  type=int, default=0)
parser.add_argument('--Train_Size',  type=int, default=500000 )
parser.add_argument('--Test_Size',  type=int, default=120000 )
parser.add_argument('--lr', type=float, default = 1e-4)
parser.add_argument('--num_heads', type=int, default = 6)
parser.add_argument('--depth', type=int, default = 4)
parser.add_argument('--ckpt_path', type=str, default = None)
parser.add_argument('--dataset_path', type=str, default = None)
parser.add_argument('--sigma_entropy', type=float, default = 1.0)
parser.add_argument('--resume_training', help='Boolean flag.', type=eval, choices=[True, False], default='False')
parser.add_argument('--debias', help='Boolean flag.', type=eval, choices=[True, False], default='False')
parser.add_argument('--dataset', type=str, default = 'Ising')
parser.add_argument('--bits', help='Boolean flag.', type=eval, choices=[True, False], default='True')


class Minde_Ising_c(pl.LightningModule):

    def __init__(self, input_size, dim_x, dim_y, lr=1e-3, mod_list=["x", "y"], use_skip=True,
                 debias=False, weighted=False, use_ema=False,
                 d=0.5, test_samples=None, gt=0.0, aes=None,
                 rows=28, depth=4, num_heads=6, map_labels=None,
                 sigma_entropy=1.0, bits=True, patch_size=2
                 ):
        super(Minde_Ising_c, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.mod_list = mod_list
        self.gt = gt
        self.weighted = weighted
        self.map_labels=map_labels[1].to(self.device), map_labels[0].to(self.device)
        self.sigma_entropy = sigma_entropy
        self.bits = bits

        """if use_skip == True:
            dim = (dim_x + dim_y)
            if dim <= 5:
                hidden_dim = 32
            elif dim <= 10:
                hidden_dim = 64
            elif dim <= 50:
                hidden_dim = 96
            else:
                hidden_dim = 128
            hidden_dim = 256
            time_dim = hidden_dim
            self.score = UnetMLP(dim=(dim_x + dim_y), init_dim=hidden_dim,
                                 dim_mults=(1, 1), time_dim=time_dim, nb_mod=2, out_dim=dim_x)"""
        
        self.sizes = [dim_x, dim_y]
        dim = np.sum(self.sizes)

        if dim <= 10:
            hidden_dim = 128
        elif dim <= 50:
            hidden_dim = 128
        elif dim <= 100:
            hidden_dim = 192
        else:
            hidden_dim = 256

        dim_m = np.max(self.sizes)
        if dim_m <= 5:
            htx = 12
        elif dim_m <= 10:
            htx = 18
        else:
            htx = 24

        self.score = DiT(input_size=input_size,
                            patch_size=patch_size,
                            in_channels=1,
                            hidden_size=htx * len(mod_list),
                            depth=4,
                            num_heads=6,
                            num_classes=31,
                            learn_sigma=False)        


        self.d = d
        self.stat = None
        self.debias = debias
        self.lr = lr
        self.use_ema = use_ema
        self.rows = rows

        self.save_hyperparameters(
            "d", "debias", "lr", "use_ema", "weighted", "dim_x", "dim_y", "gt", "rows")

        self.aes = aes
        #self.embed = nn.Linear(1, dim_y)


        self.test_samples = self.get_test_dataset(test_samples)
        self.T = torch.nn.Parameter(
            torch.FloatTensor([T0]), requires_grad=False)
        self.model_ema = EMA(self.score, decay=0.999) if use_ema else None
        self.sde = VP_SDE(importance_sampling=self.debias,
                          liklihood_weighting=False)

    def get_test_dataset(self, loader):
        X = torch.Tensor().to(self.device)
        Y = torch.Tensor().to(self.device)
        nb_lines = self.rows
        for batch in loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            X = torch.cat([X, x])
            Y = torch.cat([Y, y])
        Y=Y.type(torch.int64)

        return {
            "x": X,
            "y": Y
        }
    
    def get_modalities(self, x):
        return {
            "x": x[0],
            "y": x[1]
        }


    def training_step(self, batch, batch_idx):

        self.train()
        data = self.get_modalities(batch)

        #z = self.encode(data)
        z=data

        if self.global_step == 0:
            self.stat = {}
            #ATTENTION: it could be improper to standardize the second dimension?
            for mod in self.mod_list:
                self.stat[mod] = {
                    "mean": z[mod].float().mean(dim=0),
                    "std": z[mod].float().std(dim=0),
                }

            std = self.stat[mod]["std"].clone()
            self.stat[mod]["std"][std == 0] = 1.0

            print(self.stat)
            with open(os.path.join(self.logger.log_dir, "stat.pickle"), "wb") as f:
                pickle.dump(self.stat, f)
        #z = self.standerdize(z)

        loss = self.sde.train_step_cond(z, self.score, d=self.d).mean()
        # forward and compute loss
        self.log("loss", loss)
        return {"loss": loss}

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)

    def score_inference(self, x, t, y, std=None):
        with torch.no_grad():
            self.eval()
            if self.use_ema:
                self.model_ema.module.eval()
                return self.model_ema.module(x, t, y, std)
            else:
                return self.score(x, t, y,std)

    def validation_step(self, batch, batch_idx):
        self.eval()
        batch = self.get_modalities(batch)
    
        z=batch

        loss = self.sde.train_step_cond(
            z, self.score, d=self.d).mean()  # # forward and compute loss
        self.log("loss_test", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        #params = list(self.embed.parameters()) + list(self.score.parameters()) 

        optimizer = torch.optim.Adam(
            self.score.parameters(), lr=self.lr, amsgrad=False)
        return optimizer
        #optimizers = torch.optim.Adam(self.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma = 0.9)
        #return {"optimizer": optimizers, "lr_scheduler": scheduler, "monitor" : "loss"}


    def plot_grad_flow(self, named_parameters, my_logger=None):
        ave_grads = []
        max_grads= []
        layers = []
        i = 0
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                i += 1
                if p.grad !=None:
                    layers.append("layer_"+str(i))
                    ave_grads.append(p.grad.cpu().detach().abs().mean())
                    max_grads.append(p.grad.cpu().detach().abs().max())
                else:
                    print(n)
        plt.bar(np.arange(len(max_grads)), max_grads, lw=4, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, lw=4, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                    matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                    matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        # Save the plot as an image
        #plt.savefig("gradient_flow.png")
        #plt.close()

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        plt.close()
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        if my_logger == None:
            self.logger.experiment.add_image("grad flow", im, global_step=self.current_epoch)
        else:
            my_logger.experiment.add_image("grad flow", im, global_step=self.current_epoch)


    def plot_entropies(self, results, my_logger=None):
        ground_truth = []
        entropy_estimates = []
        temperatures = []
        for temperature in self.gt.keys():
            temperatures.append(temperature)
            ground_truth.append(self.gt[temperature])
            entropy_estimates.append(results[str(temperature)]['entropy'])
        plt.plot(temperatures,ground_truth, 'c')
        print(temperatures,ground_truth)
        print(temperatures, entropy_estimates)
        plt.plot(temperatures, entropy_estimates, 'b')
        plt.xlabel("Temperature")
        plt.ylabel("Entropy")
        #plt.title("")
        plt.grid(True)
        plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                    matplotlib.lines.Line2D([0], [0], color="b", lw=4)], ['ground_truth', 'estimates'])
        plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        plt.close()
        if my_logger == None:
            self.logger.experiment.add_image("entropies", im, global_step=self.current_epoch)
        else:
            my_logger.experiment.add_image("entropies", im, global_step=self.current_epoch)

    def log_samples(self, debias = False, my_logger=None, num_samples=8):
        if my_logger==None:
            logger=self.logger
        else:
            logger=my_logger
        test_samp_small = {
            "x": self.test_samples["x"][:num_samples].to(self.device),
            "y": self.test_samples["y"][:num_samples].to(self.device)
        }

   
        z_c=test_samp_small.copy()
        

        test_samp_small["y"] = ising_dataset.remap_values(self.map_labels ,test_samp_small["y"])
        

        log_modalities(logger, test_samp_small, [
                       "x", "y"], self.current_epoch, prefix="real/", nb_samples=8)

        

        x, y = z_c["x"], z_c["y"]

        eps=1e-5
        mods_list=list(z_c.keys())
        nb_mods = len(mods_list)

        if debias:
            #t_ = self.sde.sample_debiasing_t(
                #[x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device)
            t_ = self.sde.sample_debiasing_t(shape=(x.shape[0], 1)).to(self.device)
        else:
            t_ = torch.rand((x.shape[0], 1)
                            ).to(self.device) * (self.T - eps) + eps

        t_n = t_.expand((x.shape[0], nb_mods))

        Y, z_m, std, g, mean, mean_T, std_T = self.sde.sample(t_n, z_c, mods_list)

        std_x = std["x"]
        mean_x = mean["x"]

        #y_xc = concat_vect({
            #"x": Y["x"],
            #"y": z_c["y"]}
        #)

        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())

        t_n_x = t_n * mask_time_x + 0.0 * (1 - mask_time_x)
        t_n_c = t_n * mask_time_x + 1.0 * (1 - mask_time_x)

        with torch.no_grad():
            if debias:
                #a_xy = - self.score_inference(y_xc, t_, None,mask=mask_time_x).detach()
                a_xy = - self.score_inference(x=Y["x"], t=t_, y=z_c["y"]).detach()
            else:
                #a_xy = - self.score_inference(y_xc, t_, std_x,mask=mask_time_x).detach()
                a_xy = - self.score_inference(x=Y["x"], t=t_, y=z_c["y"], std=std_x).detach()
        
        noisy_x=Y
        
        log_modalities(logger, noisy_x, [
                       "x"], self.current_epoch, prefix="noisy/", nb_samples=8)
        
        denoised={}
        denoised["x"] = (Y["x"] + std["x"]*a_xy)/mean["x"]
        
        log_modalities(logger, denoised, [
                       "x"], self.current_epoch, prefix="denoised/", nb_samples=8)
        
        difference={}
        dim=int(math.sqrt(torch.numel(denoised["x"])/num_samples))
        difference["x"]=test_samp_small["x"]-denoised["x"].view(num_samples, 1, dim, dim)
        
        log_modalities(logger, difference, [
                       "x"], self.current_epoch, prefix="real-denoised/", nb_samples=8)

        x2= torch.randn_like(z_c["x"]).to(z_c["x"])

        if self.use_ema:
           # output_cond_0 = self.sde.modality_inpainting(score_net=self.model_ema.module,x = x1 , mask = masks[0],  subset=[0])
            output_cond_1 = self.sde.generation_c(
                score_net=self.model_ema.module, x=x2, y=z_c["y"])
        else:
          #  output_cond_0 = self.sde.modality_inpainting(score_net=self.score,x = x1 , mask = masks[0],  subset=[0])
            output_cond_1 = self.sde.generation_c(
                score_net=self.model_ema.module, x=x2, y=z_c["y"])
        
        z_c["y"] = ising_dataset.remap_values(self.map_labels,z_c["y"])

        log_modalities(logger, {"x": output_cond_1, "y":z_c["y"]}, [
                       "x", "y"], self.current_epoch, prefix="cond_1/", nb_samples=8)


    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.sde.device = self.device
    
        if self.current_epoch % 5 == 0:
            self.log_samples(debias=self.debias)

        if self.current_epoch % 5 == 0:

            #params =  list(self.embed.named_parameters()) +list(self.score.named_parameters())
            self.plot_grad_flow(named_parameters=self.score.named_parameters())


            self.test_samples["x"] = self.test_samples["x"].to(self.device)
            self.test_samples["y"] = self.test_samples["y"].to(self.device)
   
            z=self.test_samples
            temperatures=ising_dataset.remap_values(self.map_labels, self.test_samples["y"])

            r={}
            for temperature in self.gt.keys():
                #print('ground truth')
                #print(self.gt[temperature])
                embedded_temperatures=z["y"]
                data=z["x"]
                #scaled_temperature = (temperature - self.stat["y"]["mean"])/self.stat["y"]["std"]
                indexes=[i for i in range(len(temperatures)) if temperatures[i][0]==temperature]
                selected_data=data[indexes]
                selected_temperatures=embedded_temperatures[indexes]
                new_z ={"x":selected_data, "y":selected_temperatures}
                entropy = self.entropy_compute(new_z, debias=self.debias, sigma=self.sigma_entropy)
                r[str(temperature)]={'gt': self.gt[temperature],  'entropy': float(entropy.cpu().numpy())}

                self.logger.experiment.add_scalars('Estimation entropy {current_temperature}'.format(current_temperature=str(temperature)),
                                               r[str(temperature)], global_step=self.global_step)
                
            self.plot_entropies(r)

            if self.current_epoch % 5 == 0:
                with open(os.path.join(self.logger.log_dir, "results_{}.json".format(self.current_epoch)), 'w') as fp:
                    json.dump(r, fp)

    def evaluate_pretrained(self, my_logger=None) -> None:
        self.sde.device = self.device
        my_logger=my_logger
        if self.current_epoch % 5 == 0:
            self.log_samples(debias=self.debias, my_logger=my_logger)

        if self.current_epoch % 5 == 0:

            self.test_samples["x"] = self.test_samples["x"].to(self.device)
            self.test_samples["y"] = self.test_samples["y"].to(self.device)

            z=self.test_samples
            temperatures=ising_dataset.remap_values(self.map_labels, self.test_samples["y"])

            r={}
            
            for temperature in self.gt.keys():

                embedded_temperatures=z["y"]
                data=z["x"]
                indexes=[i for i in range(len(temperatures)) if temperatures[i][0]==temperature]
                selected_data=data[indexes]
                selected_temperatures=embedded_temperatures[indexes]
                new_z ={"x":selected_data, "y":selected_temperatures}

                r[str(temperature)]={}
                SIGMAS = [0.01, 0.1, 1, 10]#0.5, 1, 2, 10
                N_runs=50
                for sigma in SIGMAS:

                    r[str(temperature)][sigma] = {}

                    entropy = []
                    for i in range(N_runs):
                        entropy.append(self.entropy_compute(new_z, debias=self.debias, sigma=sigma).cpu())  #sigma=self.sigma_entropy

                    r[str(temperature)][sigma]={'gt': self.gt[temperature],  'mean': float(np.mean(entropy)),
                                                'std': float(np.std(entropy)), 'max': float(np.max(entropy)),
                                                'min': float(np.min(entropy)), 's_run': float(entropy[0])}

                for sigma in SIGMAS:
                    my_logger.experiment.add_scalars('Estimation entropy {current_temperature} {sigma}'.format(current_temperature=str(temperature), sigma = str(sigma)),
                                               r[str(temperature)][sigma], global_step=self.global_step)
            simple_r={}
            for temperature in self.gt.keys():
                simple_r[str(temperature)]={'gt': self.gt[temperature],  'entropy': r[str(temperature)][1]['mean']}

            self.plot_entropies(simple_r, my_logger=my_logger)

            if self.current_epoch % 5 == 0:
                with open(os.path.join(my_logger.log_dir, "results_{}.json".format(self.current_epoch)), 'w') as fp:
                    json.dump(r, fp)

    def get_mask(self, modalities_list_dim, subset, shape):
        mask = torch.zeros(shape)
        idx = 0
        for index_mod, dim in enumerate(modalities_list_dim):
            if index_mod in subset:
                mask[:, idx:idx + dim] = 1.0
            idx = idx + dim
        return mask
    
    def entropy_compute(self, data, debias=False, sigma=1.0, eps=1e-5):

        self.sde.device = self.device
        self.score.eval()

        x, y = data["x"], data["y"]

        mods_list = list(data.keys())
        mods_sizes = [data[key].size(1) for key in mods_list]
        nb_mods = len(mods_list)

        if debias:
            #t_ = self.sde.sample_debiasing_t(
                #[x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(self.device)
            t_ = self.sde.sample_debiasing_t(shape=(x.shape[0], 1)).to(self.device)
        else:
            t_ = torch.rand((x.shape[0], 1)
                            ).to(self.device) * (self.T - eps) + eps

        t_n = t_.expand((x.shape[0], nb_mods))

        Y, _, std, g, mean, mean_T, std_T = self.sde.sample(t_n, data, mods_list)

        std_x = std["x"]
        mean_x = mean["x"]

        #y_xc = concat_vect({
            #"x": Y["x"],
            #"y": data["y"]}
        #)
        mask_time_x = torch.tensor([1, 0]).to(self.device).expand(t_n.size())

        t_n_x = t_n * mask_time_x + 0.0 * (1 - mask_time_x)
        t_n_c = t_n * mask_time_x + 1.0 * (1 - mask_time_x)

        with torch.no_grad():
            if debias:
                #a_x = - self.score_inference(y_x, t_n_x, None).detach()
                #a_xy = - self.score_inference(y_xc, t_, None,mask=mask_time_x).detach()
                a_xy = - self.score_inference(x=Y["x"], t=t_, y=data["y"]).detach()

            else:
                #a_x = - self.score_inference(y_x, t_n_x, std_x).detach()
                #a_xy = - self.score_inference(y_xc, t_, std_x,mask=mask_time_x).detach()
                a_xy = - self.score_inference(x=Y["x"], t=t_, y=data["y"], std=std_x).detach()

        N = x.size(-1)*x.size(-2)
        M = x.size(0)
        

        chi_t_x = mean_x ** 2 * sigma ** 2 + std_x**2
        ref_score_x = (Y["x"])/chi_t_x  # was *g

        chi_T= mean_T**2*sigma**2 + std_T**2

        if debias:
            # std = std["x"][:,0].reshape(t_.shape)
            const = get_normalizing_constant((1,), T=1-eps).to(x)

            e_xc = const * 0.5 * ((a_xy + std_x * ref_score_x)**2).sum() / M

        else:
            #g = g["x"].reshape(g["x"].size(0), 1)
            g=g["x"].view(len(g["x"]), *(1,)*(len(a_xy.size())-1)).expand(a_xy.size())
            

            e_xc = 0.5 * (g**2*(a_xy + ref_score_x)**2).sum() / M


        entropy =N/2*math.log(2*math.pi*sigma**2) + (x**2).sum()/(M*2*sigma**2)- e_xc - N/2*(math.log(chi_T) - 1 + 1/chi_T)

        #print('Computation')
        #print(N/2*math.log(2*math.pi*sigma**2))
        #print((x**2).sum()/(M*2*sigma**2))
        #print(- e_xc )
        #print(- N/2*(math.log(chi_T) - 1 + 1/chi_T))
        #print(entropy)
        if self.bits:
            return entropy/(N*math.log(2))
        else:   
            return entropy/N


def get_stat(file_name):
    t = pickle.load(open(file_name, "rb"))
    for key in t.keys():
        if key != 'cat':
            t[key]['mean'] = t[key]['mean'].to("cuda")
            t[key]['std'] = t[key]['std'].to("cuda")
    return t


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    pl.seed_everything(args.seed)
    Batch_size = 64
    NUM_epoch = 500
    rows = args.rows
    dim_x = 400
    dim_y = 1
    LR = args.lr
    ckpt_path = args.ckpt_path
    depth=args.depth
    sigma_entropy = args.sigma_entropy
    num_heads =args.num_heads
    resume_training= args.resume_training
    debias = args.debias
    dataset=args.dataset
    bits = args.bits
    dataset_path = args.dataset_path

    #ground_truth = {1.0:11.854251680982795}#gaussian, divide by 4
    #ground_truth = {1.0:2.4289778942146247}#gaussian2
    
    train_l, test_l, remapping, ground_truth = ising_dataset.get_ising_dataset(batch_size=Batch_size, train_size = args.Train_Size, test_size = args.Test_Size, dataset=dataset, dataset_path=dataset_path)

    train_samp = next(iter(train_l))[0][:8,]
    test_samp = next(iter(test_l))[0][:8,]

    input_size = train_samp.size(-1)

    if dataset=='gaussian':
        patch_size=1    
    elif input_size <4:
        patch_size=1
    elif input_size%2==0:
        patch_size=2
    elif input_size%5==0:
        patch_size=5
    else:
        print('\n Unvalid patch_size\n')




    if ckpt_path==None:

        mld = Minde_Ising_c(mod_list=["x", "y"],
                    input_size=input_size,
                    dim_x=dim_x, dim_y=dim_y, lr=LR,
                    test_samples=test_l, 
                    gt=ground_truth, 
                    use_ema=True,
                    debias=debias,
                    weighted=False, 
                    d=0.5, 
                    depth=depth, 
                    num_heads=num_heads, 
                    map_labels=remapping,
                    sigma_entropy=sigma_entropy,
                    bits=bits,
                    patch_size=patch_size)
        CHECKPOINT_DIR = "runs/trained_models/mld_c_entropy/"+str(args.seed)+"/"

        tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                  name="entropy"
                                  )
        
        trainer = pl.Trainer(
            logger=tb_logger,
            check_val_every_n_epoch=10,
            accelerator='gpu',
            devices=1,
      
            max_epochs=NUM_epoch,
            default_root_dir=CHECKPOINT_DIR)

        trainer.fit(model=mld, train_dataloaders=train_l,
                val_dataloaders=test_l,)
    
    else:

        if resume_training==False:

            CHECKPOINT_DIR = "runs/trained_models/mld_c_entropy/"+str(args.seed)+"/"

            tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                  name="entropy"
                                  )
            mld = Minde_Ising_c.load_from_checkpoint(ckpt_path, input_size=input_size, dim_x=dim_x, dim_y=dim_y, lr=LR,
                      test_samples=test_l, 
                      gt=ground_truth, 
                      use_ema=True,
                      debias=debias,
                      weighted=False, 
                      d=0.5, 
                      depth=depth, 
                      num_heads=num_heads, 
                      map_labels=remapping,
                      sigma_entropy=sigma_entropy,
                      bits=bits,
                      patch_size=patch_size)
        
            mld.to("cuda")
            mld.eval()

            mld.evaluate_pretrained(my_logger=tb_logger)

            time.sleep(60)
        
        else:

            CHECKPOINT_DIR = "runs/trained_models/mld_c_entropy/"+str(args.seed)+"/"

            tb_logger = TensorBoardLogger(save_dir=CHECKPOINT_DIR,
                                  name="entropy"
                                  )
            mld = Minde_Ising_c.load_from_checkpoint(ckpt_path, input_size=input_size, dim_x=dim_x, dim_y=dim_y, lr=LR,
                      test_samples=test_l, 
                      gt=ground_truth, 
                      use_ema=True,
                      debias=debias,
                      weighted=False, 
                      d=0.5, 
                      depth=depth, 
                      num_heads=num_heads, 
                      map_labels=remapping,
                      sigma_entropy=sigma_entropy,
                      bits=bits,
                      patch_size=patch_size)        

            trainer = pl.Trainer(
                logger=tb_logger,
                check_val_every_n_epoch=10,
                accelerator='gpu',
                devices=1,
      
                max_epochs=NUM_epoch,
                default_root_dir=CHECKPOINT_DIR)

            trainer.fit(model=mld, train_dataloaders=train_l,
                val_dataloaders=test_l,)

