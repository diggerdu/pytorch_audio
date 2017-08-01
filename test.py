# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# kernel = torch.from_numpy(np.random.random_sample((1, 3))).float()
kernel = torch.from_numpy(np.ones((1, 3))).float()
# input_ = torch.from_numpy(np.random.random_sample((1, 1, 1, 6))).float()
input_ = torch.from_numpy(np.ones((1, 1, 1, 6))).float()

print(kernel)
print(input_)

model = nn.Sequential(nn.Conv2d(1, 12, (1, 3), stride=1, padding=0, bias=False))


print(model[0].weight.data)











'''
model[0].weight.data.copy_(kernel)
input_variable = Variable(input_)
out = model(input_variable)

print(out)

from numpy.linalg import inv

kernel = torch.from_numpy(np.ones((1, 3))).float()
model_trans = nn.Sequential(nn.ConvTranspose2d(1, 1, (1, 3), stride=2, padding=0, bias=False))
model_trans[0].weight.data.copy_(kernel)
re_input = model_trans(out)

print(re_input)
'''
