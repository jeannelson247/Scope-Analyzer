import numpy as np

def tool(y, w, k):
    kern = np.ones(int(w))/int(w)
    base = np.convolve(y, kern, mode='same')
    r = y - base
    sig = 1.4826*np.median(np.abs(r))
    out = y.copy()
    out[np.abs(r) > k*sig] = base[np.abs(r) > k*sig]
    return out

