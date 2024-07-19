
import torch
import random
import math

from .util import *
from .importance import *


class VP_SDE():
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 N=1000,
                 importance_sampling=True,
                 liklihood_weighting=False,
                 nb_mod=2
                 ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.liklihood_weighting = liklihood_weighting
        self.device = "cuda"
        self.nb_mod = nb_mod
        self.t_epsilon = 1e-3

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t):
        return -0.5*self.beta_t(t), torch.sqrt(self.beta_t(t))

    def marg_prob(self, t, x=None):
        # return mean std of p(x(t))
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

        log_mean_coeff = log_mean_coeff.to(self.device)

        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        if x.dim()>2:
            res1 = torch.einsum('bd, bdij -> bdij', mean, torch.ones_like(x).to(self.device))
            res2 = torch.einsum('bd, bdij -> bdij', std, torch.ones_like(x).to(self.device))
        else:
            res1=mean * torch.ones_like(x).to(self.device)
            res2=std.view(-1, 1) * torch.ones_like(x).to(self.device)
        return res1, res2

    def marg_T(self, t):
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

        log_mean_coeff = log_mean_coeff

        mean = math.exp(log_mean_coeff)
        std = math.sqrt(1 - math.exp(2 * log_mean_coeff))
        return mean, std


    def sample(self, t, data, mods_list):

        nb_mods = len(mods_list)
        self.device = t.device

        x_t_m = {}
        std_m = {}
        mean_m = {}
        z_m = {}
        g_m = {}

        for i, mod in enumerate(mods_list):
            x_mod = data[mod]

            z = torch.randn_like(x_mod.float()).to(self.device)#ADDED float
            f, g = self.sde(t[:, i])

            mean_i, std_i = self.marg_prob(
                t[:, i].view(x_mod.shape[0], 1), x_mod)

            std_m[mod] = std_i
            mean_m[mod] = mean_i
            z_m[mod] = z
            g_m[mod] = g
            x_t_m[mod] = mean_i * x_mod + std_i * z
        
        mean_T , std_T= self.marg_T(t=self.T)

        return x_t_m, z_m, std_m, g_m, mean_m, mean_T, std_T


    def train_step_cond(self, data, score_net, eps=1e-3, d=0.5):
        # data= unsequeeze_dict(data)
        #x = concat_vect(data)
        x=data["x"]

        mods_list = list(data.keys())

        nb_mods = len(mods_list)

        if self.importance_sampling:
            t = (self.sample_debiasing_t(
                shape=(x.shape[0], 1))).to(self.device)
        else:
            t = ((self.T - eps) *
                 torch.rand((x.shape[0], 1)) + eps).to(self.device)

        t_n = t.expand((x.shape[0], nb_mods))

        mask = [1, 0]

        x_t_m, z_m, std_m, g_m, mean_m, _, _ = self.sample(
            t=t_n, data=data, mods_list=mods_list)

        mask = [1, 0]
        mask_time = torch.tensor(mask).to(self.device).expand(t_n.size())
        t_n = t_n * mask_time + 1.0 * (1 - mask_time)
        
        score = - score_net(x=x_t_m["x"], t=t, y=data["y"])
        
        

        weight = 1.0
        loss = weight * torch.square(score + z_m["x"]).mean()#.sum(1, keepdim=False)
        return loss

    def sample_debiasing_t(self, shape):
        """
        non-uniform sampling of t to debias the weight std^2/g^2
        the sampling distribution is proportional to g^2/std^2 for t >= t_epsilon
        for t < t_epsilon, it's truncated
        """
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, t_epsilon=self.t_epsilon, T=self.T)
    
    
    def euler_step_c(self, x_t,x_0, t,dt, score_net, y):
             
        time = t  * torch.ones( (x_t.shape[0], 1) ).to(self.device)#, self.nb_mod
        
        
        mean,std = self.marg_prob(time,x_t)
        
       
        ## score
        with torch.no_grad():
            
            s = - score_net(x=x_t,  t=t * torch.ones( (x_t.shape[0])).to(self.device), y=y).detach()#time
            s = s /std[:,:s.size(1)]#manual division. Could pass std to score net and would obtain same result


        f,g = self.sde(t)
        ## Euler step
        if t == 0.001:
            noise = 0
        else :
            noise = torch.randn_like(s)

        x = x_t - dt*(f*x_t - (g**2) *s)  + g * torch.sqrt(dt) * noise
        
        return x  , t-dt

    def generation_c(self, score_net,x, y):
        
        t = torch.Tensor([1.0]).to(self.device)
        t_ind = 1.0
        dt = torch.Tensor( t/self.N).to(self.device)
        x_c = x    
        while t_ind>0:
            x_c,t = self.euler_step_c(x_t= x_c, x_0= x.clone(),t= t, dt = dt, score_net= score_net,y=y )
            t_ind = t_ind - dt
        return x_c
