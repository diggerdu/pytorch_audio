import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F


input = np.arange(9).reshape((3, 3)).T
input = input[np.newaxis, :, np.newaxis, :]
input = Variable(torch.from_numpy(input).float())
print(type(input))
weights = np.zeros((3, 3), np.float32)
np.fill_diagonal(weights, 1)
print(input)
weights = Variable(torch.from_numpy(weights.reshape((3, 1, 1, 3))))
output = F.conv_transpose2d(input, weights, stride=2)
print(output)




