# MINDE
Thesis code - study of diffusion models at EURECOM

Entropy Estimation for different systems of spin based on https://arxiv.org/abs/2310.09031

To reproduce  the experiments:

```
git clone https://github.com/GiuliaCornara/MINDE.git
pip install -r /content/MINDE/requirements.txt
cd /MINDE
python3 -m entropy.entropy_test_1
```

All the possible argouments are:
```
--Train_Size
--Test_Size
--lr                 #learning rate
--num_heads          #transformer parameter
--depth              #transformer parameter
--ckpt_path          #to resume training or make evaluations, the path of a checkpoint must be specified
--dataset            #choose on which dataset to carry out the experiment (Ising, XY,...)
--dataset_path       #specify the path of a custom dataset to be used
--resume_training    #if a checkpoint is given, this parameters allow to resume the training or go to evaluation
--debias             #if True, importance sampling is used
--bits               #entropy can be computed in bits or nats (by default in bits)
  
```

