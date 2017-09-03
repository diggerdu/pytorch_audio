import unittest


import time_frequency as tf
import numpy as np
import torch
from torch.autograd import Variable

class TimeFrequencyTestCase(unittest.TestCase):
    def test_istft(self):
        signal = np.arange(1024)
        input_ = np.fft.fft(signal, n=1024)
        ac = Variable(torch.from_numpy(np.real(input_[0]) * np.ones((1, 1024, 1, 1))).float())
        input_ = np.reshape(input_[1:513], (1, 1, 512, 1))
        input_real = Variable(torch.from_numpy(np.real(input_)).float())
        input_imag = Variable(torch.from_numpy(np.imag(input_)).float())



        model = tf.istft(n_fft=1024)
        output = model.forward(input_real, input_imag, ac)
        print(output.data.numpy().flatten())
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()


