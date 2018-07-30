try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

from dtcwt_slim.coeffs import biort as _biort, qshift as _qshift
from dtcwt_slim.torch.lowlevel import prep_filt
from dtcwt_slim.torch import transform_funcs as tf


class DTCWTForward(nn.Module):
    def __init__(self, C, biort='near_sym_a', qshift='qshift_a',
                 J=3, skip_hps=None):
        super().__init__()
        self.C = C
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps
        self.J = J
        h0o, g0o, h1o, g1o = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, C), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, C), False)
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        self.h0a = torch.nn.Parameter(prep_filt(h0a, C), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, C), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, C), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, C), False)

        # Create the function to do the DTCWT
        self.dtcwt_func = getattr(tf, 'xfm{J}{suff}'.format(
            J=J, suff='no_l1' if skip_hps else ''))

    def forward(self, input):
        y = self.dtcwt_func.apply(input, self.h0o, self.h1o, self.h0a,
                                  self.h0b, self.h1a, self.h1b)
        return y[0], y[1::2], y[2::2]


class DTCWTInverse(nn.Module):
    def __init__(self, C, biort='near_sym_a', qshift='qshift_a',
                 J=3, skip_hps=None):
        super().__init__()
        self.C = C
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps
        self.J = J
        h0o, g0o, h1o, g1o = _biort(biort)
        self.g0o = torch.nn.Parameter(prep_filt(g0o, C), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, C), False)
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        self.g0a = torch.nn.Parameter(prep_filt(g0a, C), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, C), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, C), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, C), False)

        # Create the function to do the DTCWT
        self.dtcwt_func = getattr(tf, 'ifm{J}{suff}'.format(
            J=J, suff='no_l1' if skip_hps else ''))

    def forward(self, yl, yhr, yhi):
        #  inputs = [yl,] + [val for pair in zip(yhr, yhi) for val in pair]
        inputs = (yl,) + yhr + yhi
        y = self.dtcwt_func.apply(*inputs, self.g0o, self.g1o, self.g0a,
                                  self.g0b, self.g1a, self.g1b)
        return y
