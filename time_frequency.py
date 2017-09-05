# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ifft(nn.Module):
    def __init__(self, n_fft=1024):
        super(ifft, self).__init__()
        assert n_fft % 2 == 0
        self.n_fft = int(n_fft)
        self.n_freq = n_freq = int(n_fft / 2)
        real_kernels, imag_kernels, self.ac_cof = _get_ifft_kernels(n_fft)
        self.real_conv = nn.Conv2d(1, n_fft, (n_freq, 1), stride=1, padding=0, bias=False)
        self.imag_conv = nn.Conv2d(1, n_fft, (n_freq, 1), stride=1, padding=0, bias=False)

        self.real_conv.weight.data.copy_(real_kernels)
        self.imag_conv.weight.data.copy_(imag_kernels)
        self.real_model = nn.Sequential(self.real_conv)
        self.imag_model = nn.Sequential(self.imag_conv)
    def forward(self, magn, phase, ac=None):
        assert magn.size()[2] == phase.size()[2] == self.n_freq
        output = self.real_model(magn) - self.imag_model(phase)
        if ac is not None:
            output = output + ac * self.ac_cof
        return output / self.n_fft






def _get_ifft_kernels(n_fft):
    n_fft = int(n_fft)
    assert n_fft % 2 == 0
    def kernel_fn(time, freq):
        return np.exp(1j * (2 * np.pi * time * freq) / 1024.)


    # kernels = np.fromfunction(kernel_fn, (int(n_fft), int(n_fft/2+1)), dtype=np.float32)


    kernels = np.zeros((1024, 513)) * 1j

    for i in range(1024):
        for j in range(513):
            kernels[i, j] = kernel_fn(i, j)



    ac_cof = float(np.real(kernels[0, 0]))


    kernels = 2 * kernels[:, 1:]
    kernels[:, -1] = kernels[:, -1] / 2.0

    real_kernels = np.real(kernels)
    imag_kernels = np.imag(kernels)




    real_kernels = torch.from_numpy(real_kernels[:, np.newaxis, :, np.newaxis])
    imag_kernels = torch.from_numpy(imag_kernels[:, np.newaxis, :, np.newaxis])
    return real_kernels, imag_kernels, ac_cof




class istft(nn.Module):
    def __init__(self, n_fft=1024, hop_length=512):
        super(istft, self).__init__()
        assert n_fft % 2 == 0
        assert hop_length < n_fft
        self.hop_length = hop_length

        self.n_fft = int(n_fft)
        self.n_freq = n_freq = int(n_fft / 2)
        self.real_kernels, self.imag_kernels, self.ac_cof = _get_istft_kernels(n_fft)

        trans_kernels = np.zeros((n_fft, n_fft), np.float32)
        np.fill_diagonal(trans_kernels, 1.)
        self.trans_kernels = Variable(torch.from_numpy(trans_kernels[:, np.newaxis, np.newaxis, :]).float())


    def forward(self, magn, phase, ac):
        assert magn.size()[2] == phase.size()[2] == self.n_freq
        n_fft = self.n_fft
        real_part = F.conv2d(magn, self.real_kernels)
        imag_part = F.conv2d(phase, self.imag_kernels)

        output = real_part - imag_part

        ac = ac.expand_as(output) * self.ac_cof
        output = output + ac
        output = output / self.n_fft

        output = F.conv_transpose2d(output, self.trans_kernels, stride=self.hop_length)
        return output


def _get_istft_kernels(n_fft):
    n_fft = int(n_fft)
    assert n_fft % 2 == 0
    def kernel_fn(time, freq):
        return np.exp(1j * (2 * np.pi * time * freq) / 1024.)


    # kernels = np.fromfunction(kernel_fn, (int(n_fft), int(n_fft/2+1)), dtype=np.float32)


    kernels = np.zeros((1024, 513)) * 1j

    for i in range(1024):
        for j in range(513):
            kernels[i, j] = kernel_fn(i, j)

    window = _hann(n_fft, sym=False)[:int(n_fft/2+1)]
   #  kernels = kernels * window
    ac_cof = float(np.real(kernels[0, 0]))



    kernels = 2 * kernels[:, 1:]
    kernels[:, -1] = kernels[:, -1] / 2.0

    real_kernels = np.real(kernels)
    imag_kernels = np.imag(kernels)


    real_kernels = Variable(torch.from_numpy(real_kernels[:, np.newaxis, :, np.newaxis]).float())
    imag_kernels = Variable(torch.from_numpy(imag_kernels[:, np.newaxis, :, np.newaxis]).float())
    return real_kernels, imag_kernels, ac_cof





class stft(nn.Module):
    def __init__(self, n_fft=1024, n_hop=512):
        super(stft, self).__init__()
        assert n_fft % 2 == 0
        self.n_fft = int(n_fft)
        self.nb_filter = int(n_fft / 2 + 1)



        real_kernels, imag_kernels = _get_stft_kernels(self.n_fft)
        real_conv = nn.Conv2d(1, self.nb_filter, (1, n_fft), stride=n_hop, bias=False)
        imag_conv = nn.Conv2d(1, self.nb_filter, (1, n_fft), stride=n_hop, bias=False)

        real_conv.weight.data.copy_(real_kernels)
        imag_conv.weight.data.copy_(imag_kernels)

        self.real_model = nn.Sequential(real_conv)
        self.imag_model = nn.Sequential(imag_conv)

    def forward(self, sample):
        return torch.stack([self.real_model(sample), self.imag_model(sample)])


def _get_stft_kernels(n_dft):
    n_dft = int(n_dft)
    assert n_dft % 2 == 0
    nb_filter = int(n_dft / 2 + 1)

    def calRealBin(freq, time):
        return np.cos((2 * np.pi * freq) / np.float32(n_dft) * time)

    def calImagBin(freq, time):
        return np.sin((2 * np.pi * freq) / np.float32(n_dft) * time)


    dft_real_kernels = np.fromfunction(calRealBin, (int(n_dft), int(n_dft)), dtype=np.float32)

    dft_imag_kernels = np.fromfunction(calImagBin, (int(n_dft), int(n_dft)), dtype=np.float32)
    '''

    w_ks = [(2 * np.pi * k) / float(n_dft) for k in range(n_dft)]
    timesteps = range(n_dft)
    dft_real_kernels = np.array([[np.cos(w_k * n) for n in timesteps] for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k * n) for n in timesteps] for w_k in w_ks])

    '''

    dft_window = _hann(n_dft, sym=False)
    # dft_window = np.hanning(1024)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)[:nb_filter]
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)[:nb_filter]
    dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]


    '''
    dft_window = _hann(n_dft, sym=False)
    # dft_window = np.hanning(n_dft)
    dft_window = np.array([dft_window]*n_dft).T

    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)

    dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    '''


    return torch.from_numpy(dft_real_kernels), torch.from_numpy(dft_imag_kernels)









def _hann(M, sym=True):
    '''[np]
    Return a Hann window.
    copied and pasted from scipy.signal.hann,
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L615
    ----------
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    '''
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
    if not sym and not odd:
        w = w[:-1]
    return w.astype(np.float32)

if __name__ == '__main__':
    signal = np.ones((4096,))
    input_ = Variable(torch.from_numpy(signal[np.newaxis, np.newaxis, np.newaxis, :]).float())
    model = stft()
    out = model(input_).data.numpy()
    real_out = out[0] + 1j * out[1]

    torch_out = np.abs(real_out[0,:,0,:])
    print(torch_out[:20])
    print(torch_out.shape)


    ## librosa
    import librosa
    librosa_out = np.abs(librosa.stft(signal, n_fft=1024, hop_length=256, center=False))

    print(librosa_out[:20])
    print(librosa_out.shape)

   # print(np.max(torch_out - np_out))






class transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, input):
        return torch.transpose(input, dim0, dim1, out=input)
