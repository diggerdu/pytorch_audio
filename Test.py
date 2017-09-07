import unittest


import time_frequence as tf
import numpy as np
import torch
from torch.autograd import Variable

import librosa


def CalSNR(ref, sig):
    ref_p = np.square(ref)
    noi_p = np.square(sig - ref).clip(min=1e-14)
    return 10 * np.log10(np.mean(ref_p / noi_p))


class TimeFrequencyTestCase(unittest.TestCase):


    '''
    def test_ifft(self):
        print('\n#########TESTING IFFT##########')
        N = 1024
        signal = np.random.random(N)
        input_ = np.fft.fft(signal, n=N)
        ac = Variable(torch.from_numpy(np.real(input_[0]) * np.ones((1, N, 1, 1))).float())
        input_ = np.reshape(input_[1:N//2+1], (1, 1, N//2, 1))
        input_real = Variable(torch.from_numpy(np.real(input_)).float())
        input_imag = Variable(torch.from_numpy(np.imag(input_)).float())



        model = tf.ifft(n_fft=N)
        output = model.forward(input_real, input_imag, ac).data.numpy().flatten()
        snr = CalSNR(signal, output)
        print("SNR:{} dB".format(snr))
        self.assertTrue(snr > 60)

        print('#########IFFT TESTED##########\n')


    '''

    def test_istft(self):

        print("###########TESTING ISTFT###########")
        signal = np.random.random(4096)
        spec = librosa.stft(signal, n_fft=1024, hop_length=512, center=False)
        magn = np.real(spec)[np.newaxis, :, np.newaxis, :]
        phase = np.imag(spec)[np.newaxis, :, np.newaxis, :]

        ac = magn[:, 0, :, :]
        magn = magn[:, 1:, :, :]
        phase = phase[:, 1:, :, :]

        magn = Variable(torch.from_numpy(magn).float())
        phase = Variable(torch.from_numpy(phase).float())
        ac = Variable(torch.from_numpy(ac).float())
        model = tf.istft(1024, 512, window="hanning")
        re_signal = model.forward(magn, phase, ac).data.numpy().flatten()

        snr = CalSNR(signal, re_signal)
        print("SNR:{} dB".format(snr))
        self.assertTrue(snr > 60)

        print("###########ISTFT TESTED###########\n")


    def test_istft_element(self):
        print("\n###########TESTING ISTFT ELEMENT###########")

        N = 1024
        signal = np.random.random(N)
        input_ = np.fft.fft(signal, n=N)

        ac = Variable(torch.from_numpy(np.real(input_[0]) * np.ones((1, 1, 1, 1))).float())
        input_ = np.reshape(input_[1:N//2+1], (1, N//2, 1, 1))
        input_ = np.repeat(input_, 4, axis=-1)
        input_real = Variable(torch.from_numpy(np.real(input_)).float())
        input_imag = Variable(torch.from_numpy(np.imag(input_)).float())


        model = tf.istft(n_fft=N, hop_length=N // 2, window="NO")
        output = model.forward(input_real, input_imag, ac).data.numpy().flatten()[:N]
        output[N//2:] = output[N//2:] - output[:N//2]


        snr = CalSNR(output, signal)
        print("SNR:{} dB".format(snr))

        self.assertTrue(snr > 60)
        print("###########ISTFT TRANSPOSE TESTED###########\n")



    def test_stft(self):
        print("\n###########TESTING STFT###########")

        N = 1024
        signal = np.random.random(4*N)
        input = Variable(torch.from_numpy(signal[np.newaxis, np.newaxis, np.newaxis, :]).float())
        stft_model = tf.stft()
        istft_model = tf.istft()

        magn, phase, ac = stft_model(input)

        re_signal = istft_model.forward(magn, phase, ac)

        re_signal = re_signal.data.numpy().flatten()
        snr = CalSNR(re_signal, signal)
        print("SNR:{} dB".format(snr))


        self.assertTrue(snr > 60)
        print("###########STFT TESTED###########\n")


if __name__ == '__main__':
    unittest.main()

