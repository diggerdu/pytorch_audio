import unittest


import time_frequency as tf
import numpy as np
import torch
from torch.autograd import Variable

import librosa

class TimeFrequencyTestCase(unittest.TestCase):
    def test_ifft(self):
        N = 1024
        signal = np.random.random(N)
        input_ = np.fft.fft(signal, n=N)
        ac = Variable(torch.from_numpy(np.real(input_[0]) * np.ones((1, N, 1, 1))).float())
        input_ = np.reshape(input_[1:N//2+1], (1, 1, N//2, 1))
        input_real = Variable(torch.from_numpy(np.real(input_)).float())
        input_imag = Variable(torch.from_numpy(np.imag(input_)).float())



        model = tf.ifft(n_fft=N)
        output = model.forward(input_real, input_imag, ac).data.numpy().flatten()
        snr = 10 * np.log10(np.mean(np.square(signal) / (np.square(output) - np.square(signal)).clip(min=1e-14)))

        print('#########IFFT RESULTS##########')
        print("SNR:{}dB".format(snr))
        self.assertTrue(snr > 60)

    '''
    def test_stft(self):
        # signal = np.ones((4096,))

        signal = np.arange(4096)
        spec = librosa.stft(signal, n_fft=1024, hop_length=512, center=False)
        magn = np.real(spec)[np.newaxis, np.newaxis, :, :]
        phase = np.imag(spec)[np.newaxis, np.newaxis, :, :]

        ac = magn[:, :, 0, :]
        magn = magn[:, :, 1:, :]
        phase = phase[:, :, 1:, :]
        print(magn.shape)

        magn = Variable(torch.from_numpy(magn).float())
        phase = Variable(torch.from_numpy(phase).float())
        #ac = Variable(torch.from_numpy(np.zeros((1,))).float())
        ac = Variable(torch.from_numpy(ac).float())
        model = tf.istft(1024, 512)
        re_signal = model.forward(magn, phase, ac).data.numpy().flatten()

        print("#############original############")
        print(signal.shape)
        print(signal[:10])
        print("#############reconstruct#########")
        print(re_signal.shape)
        print(re_signal[:10])

        self.assertTrue(False)

    '''
    def test_istft(self):
        N = 1024
        signal = np.random.random(N)
        input_ = np.fft.fft(signal, n=N)
        ac = Variable(torch.from_numpy(np.real(input_[0]) * np.ones((1, 1, 1, 1))).float())
        input_ = np.reshape(input_[1:N//2+1], (1, 1, N//2, 1))
        input_real = Variable(torch.from_numpy(np.real(input_)).float())
        input_imag = Variable(torch.from_numpy(np.imag(input_)).float())



        model = tf.istft(n_fft=N)
        output = model.forward(input_real, input_imag, ac).data.numpy().flatten()

        snr = 10 * np.log10(np.mean(np.square(signal) / (np.square(output) - np.square(signal)).clip(min=1e-14)))
        print("SNR:{}dB".format(snr))
        self.assertTrue(snr > 60)

if __name__ == '__main__':
    unittest.main()

