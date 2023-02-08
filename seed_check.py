import torch
import random
import numpy as np

torch.manual_seed(9999)
random.seed(9999)
np.random.seed(9999)
print(f'torch:{torch.rand(1)}')
print(f'random:{random.random()}')
print(f'numpy:{np.random.random()}')