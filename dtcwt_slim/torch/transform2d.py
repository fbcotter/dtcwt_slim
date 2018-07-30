try:
    import torch
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

import numpy as np
from dtcwt_slim.coeffs import biort as _biort, qshift as _qshift
#  from dtcwt_slim.torch.lowlevel import ColFilter, RowFilter, prep_filt
#  from dtcwt_slim.torch.lowlevel import ColDFilt, RowDFilt, ColIFilt, RowIFilt
from dtcwt_slim.torch.lowlevel import ColFilter, RowFilter, prep_filt
from dtcwt_slim.torch.lowlevel import colfilter, rowfilter, coldfilt, rowdfilt
from dtcwt_slim.torch.lowlevel import colifilt, rowifilt
#  from dtcwt_slim.torch import transform_templates as tt
from dtcwt_slim.torch import transform_funcs as tf
from torch.autograd import Function


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


class InvDTCWT2(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhi1, yhi2, g0o, g1o, g0a, g0b, g1a, g1b, low_only=False):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)

        # Level 2
        ll = yl
        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
        Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
        ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

        if not low_only:
            lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
            hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
            hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            Y = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)
        else:
            Y = rowfilter(colfilter(ll, h0o_t), h0o_t)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        low_only = ctx.low_only
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

class MyDTCWT2(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b, low_only=False):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.low_only = low_only
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        if low_only:
            Yhr = None
            Yhi = None
        else:
            Hi = rowfilter(input, h1o)
            LoHi = colfilter(Lo, h1o)
            HiLo = colfilter(Hi, h0o)
            HiHi = colfilter(Hi, h1o)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            Yhr = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            Yhi = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo

        # Level 2
        Lo = rowdfilt(Yl, h0b, h0a)
        LoLo = coldfilt(Lo, h0b, h0a)
        Hi = rowdfilt(Yl, h1b, h1a, highpass=True)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr2 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi2 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        return LoLo, Yhr, Yhi, Yhr2, Yhi2

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr, grad_Yhi, grad_Yhr2, grad_Yhi2):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        low_only = ctx.low_only
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.needs_input_grad[0]:
            # Level 2
            ll = grad_LoLo
            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            if not low_only:
                lh = c2q(grad_Yhr[:,:,0:6:5], grad_Yhi[:,:,0:6:5])
                hl = c2q(grad_Yhr[:,:,2:4:1], grad_Yhi[:,:,2:4:1])
                hh = c2q(grad_Yhr[:,:,1:5:3], grad_Yhi[:,:,1:5:3])
                Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
                Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
                grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)
            else:
                grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

        return (grad_input,) + (None,) * 7


class MyDTCWT1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b, low_only=False):
        ctx.save_for_backward(input, h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.low_only = low_only
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        if low_only:
            Yhr = None
            Yhi = None
        else:
            Hi = rowfilter(input, h1o)
            LoHi = colfilter(Lo, h1o)
            HiLo = colfilter(Hi, h0o)
            HiHi = colfilter(Hi, h1o)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            Yhr = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            Yhi = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo

        return Yl, Yhr, Yhi


    @staticmethod
    def backward(ctx, grad_Yl, grad_Yhr, grad_Yhi):
        input, h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        low_only = ctx.low_only
        grad_input = None
        grad_weight = None
        grad_low = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.needs_input_grad[0]:
            ll = grad_Yl
            if not low_only:
                lh = c2q(grad_Yhr[:,:,0:6:5], grad_Yhi[:,:,0:6:5])
                hl = c2q(grad_Yhr[:,:,2:4:1], grad_Yhi[:,:,2:4:1])
                hh = c2q(grad_Yhr[:,:,1:5:3], grad_Yhi[:,:,1:5:3])
                Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
                Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
                grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)
            else:
                grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

        return (grad_input,) + (None,) * 9


class DTCWTFwd2(nn.Module):
    def __init__(self, in_channels, biort='near_sym_a', qshift='qshift_a',
                 nlevels=3, skip_hps=None):
        super().__init__()
        self.in_channels = in_channels
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps
        self.nlevels = nlevels
        h0o, g0o, h1o, g1o = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, in_channels), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, in_channels), False)
        self.g0o = torch.nn.Parameter(prep_filt(g0o, in_channels), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, in_channels), False)
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        self.h0a = torch.nn.Parameter(prep_filt(h0a, in_channels), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, in_channels), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, in_channels), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, in_channels), False)
        self.g0a = torch.nn.Parameter(prep_filt(g0a, in_channels), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, in_channels), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, in_channels), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, in_channels), False)

    def forward(self, input):
        return MyDTCWT2.apply(input, self.h0o, self.h1o, self.h0a, self.h0b, self.h1a,
                              self.h1b)

class DTCWTFwd1(nn.Module):
    def __init__(self, in_channels, biort='near_sym_a', qshift='qshift_a',
                 nlevels=3, skip_hps=None):
        super().__init__()
        self.in_channels = in_channels
        self.biort = biort
        self.qshift = qshift
        self.skip_hps = skip_hps
        self.nlevels = nlevels
        h0o, g0o, h1o, g1o = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, in_channels), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, in_channels), False)
        self.g0o = torch.nn.Parameter(prep_filt(g0o, in_channels), False)
        self.g1o = torch.nn.Parameter(prep_filt(g1o, in_channels), False)
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(qshift)
        self.h0a = torch.nn.Parameter(prep_filt(h0a, in_channels), False)
        self.h0b = torch.nn.Parameter(prep_filt(h0b, in_channels), False)
        self.h1a = torch.nn.Parameter(prep_filt(h1a, in_channels), False)
        self.h1b = torch.nn.Parameter(prep_filt(h1b, in_channels), False)
        self.g0a = torch.nn.Parameter(prep_filt(g0a, in_channels), False)
        self.g0b = torch.nn.Parameter(prep_filt(g0b, in_channels), False)
        self.g1a = torch.nn.Parameter(prep_filt(g1a, in_channels), False)
        self.g1b = torch.nn.Parameter(prep_filt(g1b, in_channels), False)

    def forward(self, input):
        return MyDTCWT1.apply(input, self.h0o, self.h1o, self.h0a, self.h0b, self.h1a,
                              self.h1b)


class DTCWTForward_old(nn.Module):
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
        Yl, a, b = self.Layer0(X)
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

        return Yl, Yhr[0], Yhi[0]


class DTCWTInverse_old(nn.Module):
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


