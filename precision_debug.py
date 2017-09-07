import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


globals()['n_fft'] = 1024
globals()['freq'] = 250



signal = np.ones((1, 1, 1, 1024), dtype=np.float32)

def kernel_fn(time):
    return np.cos((2 * np.pi * time * freq) / float(n_fft))

kernels = np.fromfunction(kernel_fn, (1024,))[np.newaxis, np.newaxis, np.newaxis, :]

print(np.mean(kernels * signal))


input = Variable(torch.from_numpy(signal).double())
kernels = Variable(torch.from_numpy(kernels).double())

output = F.conv2d(input, kernels)

print(np.mean(output.data.numpy()))










#F.conv2d()
