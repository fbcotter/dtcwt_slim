import numpy as np
from dtcwt_slim.coeffs import biort as biort_, qshift as qshift_
from dtcwt_slim.numpy.lowlevel import fb


def dtcwt1d(x, biort='near_sym_a', qshift='qshift_a', J=3, nlevels=None):
    """ Perform a 1-d DTCWT on signal x, using the biort and qshift filters,
    returning both tree outputs.

    Parameters
    ----------
    x : ndarray
        Input signal to decompose
    biort : str
        Which biort filters to use
    qshift : str
        Which qshift filters to use
    J : int
        Number of levels of the transform
    nlevels : int
        Alternative to J (for compatability). Will use this if it is provided.

    Returns
    -------
    yla : ndarray
        Lowpass output from tree a
    ylb : ndarray
        Lowpass output from tree b
    yha : list(ndarray)
        Highpass outputs from tree a
    yhb : list(ndarray)
        Highpass outputs from tree b
    """
    if nlevels is not None:
        J = nlevels

    h0o, _, h1o, _ = biort_(biort)
    h0a, h0b, _, _, h1a, h1b, _, _ = qshift_(qshift)

    # Ensure they each have the correct padding
    h0a_b = np.pad(h0o[:,0], (0,2), mode='constant').astype('float32')
    h1a_b = np.pad(h1o[:,0], (0,0), mode='constant').astype('float32')

    h0b_b = np.pad(h0o[:,0], (0,0), mode='constant').astype('float32')
    h1b_b = np.pad(h1o[:,0], (0,2), mode='constant').astype('float32')

    h0a_q = np.pad(h0a[:,0], (0,0), mode='constant').astype('float32')
    h0b_q = np.pad(h0b[:,0], (0,0), mode='constant').astype('float32')

    h1a_q = np.pad(h1a[:,0], (0,0), mode='constant').astype('float32')
    h1b_q = np.pad(h1b[:,0], (0,0), mode='constant').astype('float32')

    yla, yha = fb(x, h0a_b, h1a_b, h0a_q, h1a_q, J=J)
    ylb, yhb = fb(x, h0b_b, h1b_b, h0b_q, h1b_q, J=J)

    return yla, ylb, yha, yhb


def dtcwt1d_cmplx(x, biort='near_sym_a', qshift='qshift_a', J=3, nlevels=None):
    """ Perform a 1-d DTCWT on signal x, using the biort and qshift filters,
    returning complex results.

    Parameters
    ----------
    x : ndarray
        Input signal to decompose
    biort : str
        Which biort filters to use
    qshift : str
        Which qshift filters to use
    J : int
        Number of levels of the transform
    nlevels : int
        Alternative to J (for compatability). Will use this if it is provided.

    Returns
    -------
    yl : ndarray
        Interleaved results from yla and ylb. yla takes even samples and ylb
        takes odd.
    yh : list(ndarray)
        Complex results from yha and yhb. yha is the real part and yhb is the
        imaginary.
    """
    yla, ylb, yha, yhb = dtcwt1d(x, biort, qshift, J, nlevels)
    yl = np.stack((ylb, yla), axis=1).reshape((-1))
    yh = [yha[i] + 1j*yhb[i] for i in range(len(yha))]
    return yl, yh
