# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable



class stft(nn.Module):
    def __init__(self, n_fft=1024, n_hop=256):
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

'''
class ISTFT(nn.Module):
    def __init__(self):
        pass
'''


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






