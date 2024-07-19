import numpy as np
import os

Ts = np.linspace(0.1, 1.5, 32)
#Tss = ['{:.2f}'.format(T) for T in Ts]
print(Ts)
#print(Tss)
for T in Ts:
    configs=np.load(os.getcwd() + '\\minde\\entropy\\xy_model_configurations\\configurations_{}.npy'.format(str(round(T, 2))))
    np.savez_compressed('configurations_{}'.format(str(round(T, 2))), configs)