# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.signal


class ifft(nn.Module):
    def __init__(self, nfft=1024):
        super(ifft, self).__init__()
        assert nfft % 2 == 0
        self.nfft = int(nfft)
        self.n_freq = n_freq = int(nfft / 2)
        real_kernels, imag_kernels, self.ac_cof = _get_ifft_kernels(nfft)
        self.real_conv = nn.Conv2d(1, nfft, (n_freq, 1), stride=1, padding=0, bias=False)
        self.imag_conv = nn.Conv2d(1, nfft, (n_freq, 1), stride=1, padding=0, bias=False)

        self.real_conv.weight.data.copy_(real_kernels)
        self.imag_conv.weight.data.copy_(imag_kernels)
        self.real_model = nn.Sequential(self.real_conv)
        self.imag_model = nn.Sequential(self.imag_conv)
    def forward(self, magn, phase, ac=None):
        assert magn.size()[2] == phase.size()[2] == self.n_freq
        output = self.real_model(magn) - self.imag_model(phase)
        if ac is not None:
            output = output + ac * self.ac_cof
        return output / self.nfft






def _get_ifft_kernels(nfft):
    nfft = int(nfft)
    assert nfft % 2 == 0
    def kernel_fn(time, freq):
        return np.exp(1j * (2 * np.pi * time * freq) / 1024.)


    kernels = np.fromfunction(kernel_fn, (int(nfft), int(nfft/2+1)), dtype=np.float64)


    kernels = np.zeros((1024, 513)) * 1j

    '''
    for i in range(1024):
        for j in range(513):
            kernels[i, j] = kernel_fn(i, j)
    '''



    ac_cof = float(np.real(kernels[0, 0]))


    kernels = 2 * kernels[:, 1:]
    kernels[:, -1] = kernels[:, -1] / 2.0

    real_kernels = np.real(kernels)
    imag_kernels = np.imag(kernels)




    real_kernels = torch.from_numpy(real_kernels[:, np.newaxis, :, np.newaxis])
    imag_kernels = torch.from_numpy(imag_kernels[:, np.newaxis, :, np.newaxis])
    return real_kernels, imag_kernels, ac_cof




class istft(nn.Module):
    def __init__(self, nfft=1024, hop_length=512):
        super(istft, self).__init__()
        assert nfft % 2 == 0
        assert hop_length <= nfft
        self.hop_length = hop_length

        self.nfft = int(nfft)
        self.n_freq = n_freq = int(nfft / 2)
        self.real_kernels, self.imag_kernels, self.ac_cof = _get_istft_kernels(nfft, window)

        trans_kernels = np.zeros((nfft, nfft), np.float64)
        win_cof = np.ones((nfft, ), dtype=np.float64)
        np.fill_diagonal(trans_kernels, win_cof)

        self.trans_kernels = nn.Parameter(torch.from_numpy(trans_kernels[:, np.newaxis, np.newaxis, :]).float())

    def forward(self, magn, phase, ac):
        print('istft_debug', magn.size())
        print('istft_debug', phase.size())
        assert magn.size()[2] == phase.size()[2] == self.n_freq
        nfft = self.nfft

        # complex conjugate
        phase = -1. * phase
        real_part = F.conv2d(magn, self.real_kernels)
        imag_part = F.conv2d(phase, self.imag_kernels)

        output = real_part - imag_part

        print('stft forward debug', output.size())

        ac = ac.unsqueeze(1)
        print('stft forward debug', ac.size())
        ac = float(self.ac_cof) * ac.expand_as(output)
        output = output + ac
        output = output / float(self.nfft)

        print('stft forward debug', output.size())
        output = F.conv_transpose2d(output, self.trans_kernels, stride=self.hop_length)
        output = output.squeeze(1)
        output = output.squeeze(1)
        print('stft forward debug output', output.size())
        return output

def _get_istft_kernels(nfft, window):
    nfft = int(nfft)
    assert nfft % 2 == 0
    def kernel_fn(time, freq):
        return np.exp(1j * (2 * np.pi * time * freq) / nfft)
    kernels = np.fromfunction(kernel_fn, (int(nfft), int(nfft/2+1)), dtype=np.float64)

    ac_cof = float(np.real(kernels[0, 0]))

    kernels = 2 * kernels[:, 1:]
    kernels[:, -1] = kernels[:, -1] / 2.0

    real_kernels = np.real(kernels)
    imag_kernels = np.imag(kernels)

    real_kernels = nn.Parameter(torch.from_numpy(real_kernels[:, np.newaxis, :, np.newaxis]).float())
    imag_kernels = nn.Parameter(torch.from_numpy(imag_kernels[:, np.newaxis, :, np.newaxis]).float())
    return real_kernels, imag_kernels, ac_cof



class stft(nn.Module):
    def __init__(self, nfft=1024, hop_length=512, window="hanning"):
        super(stft, self).__init__()
        assert nfft % 2 == 0

        self.hop_length = hop_length
        self.n_freq = n_freq = nfft//2 + 1

        self.real_kernels, self.imag_kernels = _get_stft_kernels(nfft, window)

    def forward(self, sample):
        sample = sample.unsqueeze(1)
        sample = sample.unsqueeze(1)

        magn = F.conv2d(sample, self.real_kernels, stride=self.hop_length)
        phase = F.conv2d(sample, self.imag_kernels, stride=self.hop_length)

        magn = magn.permute(0, 2, 1, 3)
        phase = phase.permute(0, 2, 1, 3)

        # complex conjugate
        phase = -1 * phase[:,:,1:,:]
        ac = magn[:,:,0,:]
        magn = magn[:,:,1:,:]
        return magn, phase, ac


def _get_stft_kernels(nfft, window):
    nfft = int(nfft)
    assert nfft % 2 == 0

    def kernel_fn(freq, time):
        return np.exp(-1j * (2 * np.pi * time * freq) / float(nfft))

    kernels = np.fromfunction(kernel_fn, (nfft//2+1, nfft), dtype=np.float64)

    if window == "hanning":
        win_cof = scipy.signal.get_window("hanning", nfft)[np.newaxis, :]
    else:
        win_cof = np.ones((1, nfft), dtype=np.float64)

    kernels = kernels[:, np.newaxis, np.newaxis, :] * win_cof

    real_kernels = nn.Parameter(torch.from_numpy(np.real(kernels)).float())
    imag_kernels = nn.Parameter(torch.from_numpy(np.imag(kernels)).float())

    return real_kernels, imag_kernels






if __name__ == '__main__':
    # signal = np.random.random(4096)
    # signal = np.arange(4096)
    signal = np.ones((4096, ))
    input_ = Variable(torch.from_numpy(signal[np.newaxis, np.newaxis, np.newaxis, :]).float())
    model = stft(nfft=1024, hop_length=512, window="hanning")
    magn, phase, ac = model(input_)
    magn = magn.data.numpy().squeeze(axis=(0, 2))


    print("#################torch_audio##################")
    print(magn[200:210, 3])
    print(magn.shape)


    ## librosa
    import librosa
    librosa_out = librosa.stft(signal, nfft=1024, hop_length=512, center=False)

    print("#################librosa##################")
    print(np.real(librosa_out)[201:211, 3])
    print(librosa_out.shape)

   # print(np.max(torch_out - np_out))

'''

if __name__ == '__main__':
    signal = 1000 * np.ones((1024,), dtype=np.float32)
    input_ = Variable(torch.from_numpy(signal[np.newaxis, np.newaxis, np.newaxis, :]).double())
    model = stft(nfft=1024, hop_length=512, window="NO")
    magn, phase, ac = model(input_)
    magn = magn.data.numpy().squeeze(axis=(0, 2)).flatten()

    np_spec = np.fft.fft(signal)

    print(magn[250:260])
    print(np.real(np_spec[251:261]))

    print(ac)
    print(np.real(np_spec[0]))]
'''
