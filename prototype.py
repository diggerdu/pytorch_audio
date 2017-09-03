import numpy as np



N = 1024
a = np.arange(N)
s = np.fft.fft(a, n=N)


X = list()




debug_K = list()
for m in range(N):
    debug = list()
    ans = list()
    for k in range(0, int(N/2)+1):
        kernel = np.exp(1j * (2 * np.pi / N) * k * m)
        debug.append(kernel)
        #ans.append(np.real(s[k]) * np.real(kernel) - np.imag(s[k]) * np.imag(kernel))
    #X.append(ans[-1] + 2 * sum(ans[1:-1]))
    debug_K.append(debug)

#X = np.array(X)



debug_K = np.array(debug_K)
print(debug_K.shape)
print(debug_K)

np.save('correct', debug_K)

#print(X[:20] / 1024.0)
