import unittest


import time_frequency as tf
import numpy as np
import torch
from torch.autograd import Variable

class TimeFrequencyTestCase(unittest.TestCase):
    def test_istft(self):
        N = 1024
        signal = np.random.random(N)
        input_ = np.fft.fft(signal, n=N)
        ac = Variable(torch.from_numpy(np.real(input_[0]) * np.ones((1, N, 1, 1))).float())
        input_ = np.reshape(input_[1:N//2+1], (1, 1, N//2, 1))
        input_real = Variable(torch.from_numpy(np.real(input_)).float())
        input_imag = Variable(torch.from_numpy(np.imag(input_)).float())



        model = tf.istft(n_fft=N)
        output = model.forward(input_real, input_imag, ac)
        result = np.allclose(output.data.numpy().flatten(), signal, rtol=1e-4)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()


