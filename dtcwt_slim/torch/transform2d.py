try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

import numpy as np
from dtcwt_slim.coeffs import biort as _biort, qshift as _qshift
from dtcwt_slim.torch.lowlevel import ColFilter, RowFilter, prep_filt
from dtcwt_slim.torch.lowlevel import ColDFilt, RowDFilt, ColIFilt, RowIFilt


class DTCWTForward2(nn.Module):
    """ Module to perform a DTCWT forward transform
    """
    def __init__(self, in_channels, biort='near_sym_a', qshift='qshift_a',
                 skip_hps=None):
        super().__init__()
        self.in_channels = in_channels
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps

        if skip_hps is None:
            skip_hps = [False,] * 2

        # Use the bandpass layers if the filters are bandpasses
        if biort.endswith('_bp') and qshift.endswith('_bp'):
            self.Layer1 = DTCWTLayerBiort_bp(in_channels, biort, skip_hps[0])
            self.Layer2 = DTCWTLayerQshift_bp(
                in_channels, qshift, skip_hps[1])
        else:
            self.Layer1 = DTCWTLayerBiort(in_channels, biort, skip_hps[0])
            self.Layer2 = DTCWTLayerQshift(
                in_channels, qshift, skip_hps[1])

    def forward(self, X):
        # Need to make sure the image is even in size
        r, c = X.shape[2:]
        if r % 2 != 0:
            # Repeat the bottom row
            xe = np.arange(0,r+1)
            xe[-1] = xe[-2]
            X = X[:,:,xe]
        if c % 2 != 0:
            # Repeat the last col
            xe = np.arange(0,c+1)
            xe[-1] = xe[-2]
            X = X[:,:,:,xe]

        Yhr = []
        Yhi = []
        Yl, a, b = self.Layer1(X)
        Yhr.append(a)
        Yhi.append(b)
        # Need to make sure the image is divisible by 4
        r, c = Yl.shape[2:]
        if r % 4 != 0:
            # Repeat the bottom row
            xe = np.arange(-1,r+1)
            xe[0], xe[-1] = xe[1], xe[-2]
            Yl = Yl[:,:,xe]
        if c % 4 != 0:
            # Repeat the last col
            xe = np.arange(-1,c+1)
            xe[0], xe[-1] = xe[1], xe[-2]
            Yl = Yl[:,:,:,xe]
        Yl, a, b = self.Layer2(Yl)
        Yhr.append(a)
        Yhi.append(b)

        return Yl, Yhr, Yhi


class DTCWTForward(nn.Module):
    """ Module to perform a DTCWT forward transform
    """
    def __init__(self, in_channels, biort='near_sym_a', qshift='qshift_a',
                 nlevels=3, skip_hps=None):
        super().__init__()
        self.in_channels = in_channels
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps
        self.nlevels = nlevels

        if skip_hps is None:
            skip_hps = [False,] * 3

        # Use the bandpass layers if the filters are bandpasses
        if biort.endswith('_bp') and qshift.endswith('_bp'):
            self.Layer0 = DTCWTLayerBiort_bp(in_channels)
            for i in range(1,nlevels):
                setattr(self, 'Layer{}'.format(i), DTCWTLayerQshift_bp(
                    in_channels))
        else:
            self.Layer0 = DTCWTLayerBiort(in_channels, biort, skip_hps[0])
            for i in range(1,nlevels):
                setattr(self, 'Layer{}'.format(i), DTCWTLayerQshift(
                    in_channels, qshift, skip_hps[i]))

    def forward(self, X):
        # Need to make sure the image is even in size
        r, c = X.shape[2:]
        if r % 2 != 0:
            # Repeat the bottom row
            xe = np.arange(0,r+1)
            xe[-1] = xe[-2]
            X = X[:,:,xe]
        if c % 2 != 0:
            # Repeat the last col
            xe = np.arange(0,c+1)
            xe[-1] = xe[-2]
            X = X[:,:,:,xe]

        Yhr = []
        Yhi = []
        Yl, a, b = self.Layer1(X)
        Yhr.append(a)
        Yhi.append(b)
        # Need to make sure the image is divisible by 4
        for i in range(1, self.nlevels):
            r, c = Yl.shape[2:]
            if r % 4 != 0:
                # Repeat the bottom row
                xe = np.arange(-1,r+1)
                xe[0], xe[-1] = xe[1], xe[-2]
                Yl = Yl[:,:,xe]
            if c % 4 != 0:
                # Repeat the last col
                xe = np.arange(-1,c+1)
                xe[0], xe[-1] = xe[1], xe[-2]
                Yl = Yl[:,:,:,xe]
            Layer = getattr(self, 'Layer{}'.format(i))
            Yl, a, b = Layer(Yl)
            Yhr.append(a)
            Yhi.append(b)

        return Yl, Yhr, Yhi


class DTCWTInverse(nn.Module):
    """ Module to perform a DTCWT forward transform
    """
    def __init__(self, in_channels, biort='near_sym_a', qshift='qshift_a',
                 nlevels=2, skip_hps=None):
        self.in_channels = in_channels
        self.biort = biort
        self.qshift = qshift
        self.nlevels = nlevels
        self.skip_hps = skip_hps

        if skip_hps is None:
            skip_hps = [False,] * nlevels

        # Use the bandpass layers if the filters are bandpasses
        if biort.endswith('_bp') and qshift.endswith('_bp'):
            self.Layer1 = DTCWTLayerBiort_bp(in_channels, biort, skip_hps[0])
            self.Layer2plus = [DTCWTLayerQshift_bp(
                in_channels, qshift, skip_hps[i]) for i in range(1,nlevels)]
        else:
            self.Layer1 = DTCWTLayerBiort(in_channels, biort, skip_hps[0])
            self.Layer2plus = [DTCWTLayerQshift(
                in_channels, qshift, skip_hps[i]) for i in range(1,nlevels)]


class DTCWTLayerBiort(nn.Module):
    def __init__(self, in_channels, coeffs, low_only=False):
        super().__init__()
        self.in_channels = in_channels
        self.coeffs = coeffs
        self.low_only = low_only
        self.biort = _biort(coeffs)
        assert len(self.biort) == 4
        h0o, _, h1o, _ = self.biort
        self.h0o = torch.nn.Parameter(prep_filt(h0o, in_channels), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, in_channels), False)

        self.lowrow = RowFilter(self.h0o)
        self.lowcol = ColFilter(self.h0o)
        if not low_only:
            self.highrow = RowFilter(self.h1o)
            self.highcol = ColFilter(self.h1o)

    def forward(self, X):
        Lo = self.lowrow(X)
        LoLo = self.lowcol(Lo)
        if not self.low_only:
            Hi = self.highrow(X)
            LoHi = self.highcol(Lo)
            HiLo = self.lowcol(Hi)
            HiHi = self.highcol(Hi)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            Yhr = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            Yhi = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)
            return LoLo, Yhr, Yhi
        else:
            return LoLo, None, None


class DTCWTLayerBiort_bp(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.biort = _biort('near_sym_b_bp')
        h0o, _, h1o, _, h2o, _ = self.biort
        self.h0o = torch.nn.Parameter(prep_filt(h0o, in_channels), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, in_channels), False)
        self.h2o = torch.nn.Parameter(prep_filt(h2o, in_channels), False)

        self.lowrow = RowFilter(self.h0o.transpose(2,3))
        self.lowcol = ColFilter(self.h0o)
        self.midrow = RowFilter(self.h2o.transpose(2,3))
        self.midcol = ColFilter(self.h2o)
        self.highrow = RowFilter(self.h1o.transpose(2,3))
        self.highcol = ColFilter(self.h1o)

    def forward(self, X):
        Lo = self.lowrow(X)
        LoLo = self.lowcol(Lo)
        Hi = self.highrow(X)
        LoHi = self.highcol(Lo)
        HiLo = self.lowcol(Hi)
        Mid = self.midrow(X)
        MidMid = self.midcol(Mid)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(MidMid)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)
        return LoLo, Yhr, Yhi


class DTCWTLayerBiort_Inv(nn.Module):
    def __init__(self, in_channels, coeffs, low_only=False):
        super().__init__()
        self.in_channels = in_channels
        self.coeffs = coeffs
        self.low_only = low_only
        self.biort = _biort(coeffs)
        assert len(self.biort) == 4
        _, g0o, _, g1o = self.biort
        self.g0o = torch.nn.Parameter(prep_filt(g0o, in_channels), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, in_channels), False)

        self.lowrow = RowFilter(self.g0o)
        self.lowcol = ColFilter(self.g0o)
        if not low_only:
            self.highrow = RowFilter(self.g1o)
            self.highcol = ColFilter(self.g1o)

    def forward(self, Yl, Yhr, Yhi):
        if self.low_only:
            y = self.lowcol(self.lowrow(Yl))
        else:
            lh = c2q(Yhr[:,:,0:6:5], Yhi[:,:,0:6:5])
            hl = c2q(Yhr[:,:,2:4:1], Yhi[:,:,2:4:1])
            hh = c2q(Yhr[:,:,1:5:3], Yhi[:,:,1:5:3])
            y1 = self.lowrow(Yl) + self.highrow(hl)
            y2 = self.lowrow(lh) + self.highrow(hh)
            y = self.lowcol(y1) + self.highcol(y2)

        return y


class DTCWTLayerBiort_bpInv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.biort = _biort('near_sym_b_bp')
        assert len(self.biort) == 6
        _, g0o, _, g1o, _, g2o = self.biort
        self.g0o = torch.nn.Parameter(prep_filt(g0o, in_channels), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, in_channels), False)
        self.g2o = torch.nn.Parameter(prep_filt(g2o, in_channels), False)

        self.lowrow = RowFilter(self.g0o)
        self.lowcol = ColFilter(self.g0o)
        self.midrow = RowFilter(self.g2o)
        self.midcol = ColFilter(self.g2o)
        self.highrow = RowFilter(self.g1o)
        self.highcol = ColFilter(self.g1o)

    def forward(self, Yl, Yhr, Yhi):
        lh = c2q(Yhr[:,:,0:6:5], Yhi[:,:,0:6:5])
        hl = c2q(Yhr[:,:,2:4:1], Yhi[:,:,2:4:1])
        hh = c2q(Yhr[:,:,1:5:3], Yhi[:,:,1:5:3])
        y1 = self.lowcol(Yl) + self.highcol(lh)
        y2 = self.lowcol(hl)
        y2bp = self.midcol(hh)
        y = self.lowrow(y1) + self.highrow(y2) + self.midrow(y2bp)

        return y


class DTCWTLayerQshift(nn.Module):
    def __init__(self, in_channels, coeffs, low_only=False):
        super().__init__()
        self.in_channels = in_channels
        self.coeffs = coeffs
        self.low_only = low_only
        self.qshift = _qshift(coeffs)
        assert len(self.qshift) == 8
        h0a, h0b, _, _, h1a, h1b, _, _ = self.qshift
        self.h0a = torch.nn.Parameter(prep_filt(h0a, in_channels), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, in_channels), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, in_channels), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, in_channels), False)

        self.lowrow = RowDFilt(self.h0b, self.h0a)
        self.lowcol = ColDFilt(self.h0b, self.h0a)
        if not low_only:
            self.highrow = RowDFilt(self.h1b, self.h1a)
            self.highcol = ColDFilt(self.h1b, self.h1a)

    def forward(self, X):
        Lo = self.lowrow(X)
        LoLo = self.lowcol(Lo)
        if not self.low_only:
            Hi = self.highrow(X)
            LoHi = self.highcol(Lo)
            HiLo = self.lowcol(Hi)
            HiHi = self.highcol(Hi)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            Yhr = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            Yhi = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)
            return LoLo, Yhr, Yhi
        else:
            return LoLo, None, None


class DTCWTLayerQshift_bp(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.qshift = _qshift('qshift_b_bp')
        assert len(self.qshift) == 12
        h0a, h0b, _, _, h1a, h1b, _, _, h2a, h2b, _, _ = self.qshift
        self.h0a = torch.nn.Parameter(prep_filt(h0a, in_channels), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, in_channels), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, in_channels), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, in_channels), False)
        self.h2a = torch.nn.Parameter(prep_filt(h2a, in_channels), False)
        self.h2b = torch.nn.Parameter(prep_filt(h2b, in_channels), False)

        self.lowrow = RowDFilt(self.h0b, self.h0a)
        self.lowcol = ColDFilt(self.h0b, self.h0a)
        self.midrow = RowDFilt(self.h2b, self.h2a)
        self.midcol = ColDFilt(self.h2b, self.h2a)
        self.highrow = RowDFilt(self.h1b, self.h1a)
        self.highcol = ColDFilt(self.h1b, self.h1a)

    def forward(self, X):
        Lo = self.lowrow(X)
        LoLo = self.lowcol(Lo)
        Hi = self.highrow(X)
        LoHi = self.highcol(Lo)
        HiLo = self.lowcol(Hi)
        Mid = self.midrow(X)
        MidMid = self.midcol(Mid)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(MidMid)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)
        return LoLo, Yhr, Yhi


class DTCWTLayerQshift_Inv(nn.Module):
    def __init__(self, in_channels, coeffs, low_only=False):
        super().__init__()
        self.in_channels = in_channels
        self.coeffs = coeffs
        self.low_only = low_only
        self.qshift = _qshift(coeffs)
        assert len(self.qshift) == 8
        _, _, g0a, g0b, _, _, g1a, g1b = self.qshift
        self.g0a = torch.nn.Parameter(prep_filt(g0a, in_channels), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, in_channels), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, in_channels), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, in_channels), False)

        self.lowrow = RowIFilt(self.g0b, self.g0a)
        self.lowcol = ColIFilt(self.g0b, self.g0a)
        if not low_only:
            self.highrow = RowIFilt(self.g1b, self.g1a)
            self.highcol = ColIFilt(self.g1b, self.g1a)

    def forward(self, Yl, Yhr=None, Yhi=None):
        if self.low_only:
            y = self.lowcol(self.lowrow(Yl))
        else:
            lh = c2q(Yhr[:,:,0:6:5], Yhi[:,:,0:6:5])
            hl = c2q(Yhr[:,:,2:4:1], Yhi[:,:,2:4:1])
            hh = c2q(Yhr[:,:,1:5:3], Yhi[:,:,1:5:3])
            y1 = self.lowrow(Yl) + self.highrow(hl)
            y2 = self.lowrow(lh) + self.highrow(hh)
            y = self.lowcol(y1) + self.highcol(y2)

        return y


class DTCWTLayerQshift_bpInv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.qshift = _qshift('qshift_b_bp')
        assert len(self.qshift) == 12
        _, _, g0a, g0b, _, _, g1a, g1b, _, _, g2a, g2b = self.qshift
        self.g0a = torch.nn.Parameter(prep_filt(g0a, in_channels), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, in_channels), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, in_channels), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, in_channels), False)
        self.g2a = torch.nn.Parameter(prep_filt(g2a, in_channels), False)
        self.g2b = torch.nn.Parameter(prep_filt(g2b, in_channels), False)

        self.lowrow = RowIFilt(self.g0b, self.g0a)
        self.lowcol = ColIFilt(self.g0b, self.g0a)
        self.midrow = RowIFilt(self.g2b, self.g2a)
        self.midcol = ColIFilt(self.g2b, self.g2a)
        self.highrow = RowIFilt(self.g1b, self.g1a)
        self.highcol = ColIFilt(self.g1b, self.g1a)

    def forward(self, Yl, Yhr=None, Yhi=None):
        lh = c2q(Yhr[:,:,0:6:5], Yhi[:,:,0:6:5])
        hl = c2q(Yhr[:,:,2:4:1], Yhi[:,:,2:4:1])
        hh = c2q(Yhr[:,:,1:5:3], Yhi[:,:,1:5:3])
        y1 = self.lowrow(Yl) + self.highrow(hl)
        y2 = self.lowrow(lh)
        y2bp = self.midrow(hh)
        y = self.lowcol(y1) + self.highcol(y2) + self.midcol(y2bp)

        return y


def q2c(y):
    """
    Convert from quads in y to complex numbers in z.
    """

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d
    # Combine (a,b) and (d,c) to form two complex subimages.
    y = y/np.sqrt(2)
    a, b = y[:,:, 0::2, 0::2], y[:,:, 0::2, 1::2]
    c, d = y[:,:, 1::2, 0::2], y[:,:, 1::2, 1::2]

    return (a-d, b+c, a+d, b-c)


def c2q(w_r, w_i):
    """
    Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 highpasses
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """

    # Input has shape [batch, ch, 2 r, c]
    ch, _, r, c = w_r.shape[1:]
    w_r = w_r/np.sqrt(2)
    w_i = w_i/np.sqrt(2)
    # shape will be:
    #   x1   x2
    #   x3   x4
    x1 = w_r[:,:,0] + w_r[:,:,1]
    x2 = w_i[:,:,0] + w_i[:,:,1]
    x3 = w_i[:,:,0] - w_i[:,:,1]
    x4 = -w_r[:,:,0] + w_r[:,:,1]

    # Stack 2 inputs of shape [batch, ch, r, c] to [batch, ch, r, 2, c]
    x_rows1 = torch.stack((x1, x3), dim=-2)
    # Reshaping interleaves the results
    x_rows1 = x_rows1.view(-1, ch, 2*r, c)
    # Do the same for the even columns
    x_rows2 = torch.stack((x2, x4), dim=-2)
    x_rows2 = x_rows2.view(-1, ch, 2*r, c)

    # Stack the two [batch, ch, 2*r, c] tensors to [batch, ch, 2*r, c, 2]
    x_cols = torch.stack((x_rows1, x_rows2), dim=-1)
    y = x_cols.view(-1, ch, 2*r, 2*c)

    return y
