from scipy.signal import convolve2d, convolve
import numpy as np
from dtcwt_slim.utils import reflect


def fb(x, h0o, h1o, h0q, h1q, J=3, padding='symm'):
    """ This function performs one tree of the forward DTCWT

    If symmetric padding is used, the initial signal we be padded so that each
    of the downsampled versions have enough padding for valid convolution (i.e.)
    more layers = wider support = more initial padding. After being used to
    calculate the next scale, the extra coeffs are discarded.  This does not
    need to be done for zero padding.

    It currently is quite inefficient (downsamples after convolving), but is
    simple to follow and matches the tree diagram nicely.

    Parameters
    ----------
    x : ndarray
        1d input to decompose
    h0o : ndarray
        first level lowpass fb
    h1o : ndarray
        first level highpass fb
    h0q : ndarray
        second+ level lowpass fb
    h1q : ndarray
        second+ level highpass fb
    padding : str
        Padding style - 'symm' or 'zero' (zero is slightly more efficient)
    """
    # Calculate padding requirements
    h0o_half = h0o.shape[0] // 2
    h0o_even = (h0o.shape[0] + 1) % 2
    h1o_half = h1o.shape[0] // 2
    h1o_even = (h1o.shape[0] + 1) % 2
    h0q_half = h0q.shape[0] // 2
    h0q_even = (h0q.shape[0] + 1) % 2
    h1q_half = h1q.shape[0] // 2
    h1q_even = (h1q.shape[0] + 1) % 2

    assert h0q_even == h1q_even == 1, "Q-shift filters should be even length"
    assert h0q_half == h1q_half, "Q-shift lp & hp should have the same length"

    c = x.shape[0]

    # Do the highpass filtering first
    before = h1o_half - h1o_even
    after = h1o_half
    xe = reflect(np.arange(-before, c + after, dtype='int32'), -0.5, c-0.5)
    x1 = convolve(x[xe], h1o, mode='valid', method='direct')[::2]

    # Do the lowpass filtering
    before = h0o_half - h0o_even
    after = h0o_half
    before_pad = [0,] * J
    after_pad = [0,] * J
    for j in range(1, J):
        before += 2**j * (h0q_half - h0q_even)
        after += 2**j * h0q_half
        for i in range(j):
            before_pad[i] += 2**(j-i-1) * (h0q_half - h0q_even)
            after_pad[i] += 2**(j-i-1) * h0q_half

    xe = reflect(np.arange(-before, c + after, dtype='int32'), -0.5, c-0.5)
    x0 = convolve(x[xe], h0o, mode='valid', method='direct')[0::2]

    x01 = []
    for j in range(1, J):
        h = convolve(x0, h1q, mode='valid', method='direct')[::2]
        x01.append(h[before_pad[j]:h.shape[0]-after_pad[j]])
        x0 = convolve(x0, h0q, mode='valid', method='direct')[0::2]

    return x0, [x1,] + x01
