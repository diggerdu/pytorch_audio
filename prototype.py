import numpy as np



N = 1024
a = np.arange(N)
s = np.fft.fft(a, n=N)



X = list()

for m in range(N):
    ans = list()
    for k in range(int(N/2)+1):
        kernel = np.exp(1j * (2 * np.pi / N) * k * m)
        ans.append(np.real(s[k]) * np.real(kernel) - np.imag(s[k]) * np.imag(kernel))
    X.append(ans[0] + ans[-1] + 2 * sum(ans[1:-1]))

X = np.array(X)


print(X[:20] / 1024.0)





