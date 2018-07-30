import torch
from torch.autograd import Function
from dtcwt_slim.torch.lowlevel import ColFilter as colfilter, RowFilter as rowfilter
from dtcwt_slim.torch.lowlevel import coldfilt, rowdfilt, prep_filt
from dtcwt_slim.torch.lowlevel import colifilt, rowifilt, q2c, c2q

class ifm1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhi1, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhi1 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            Hi = rowfilter(grad_y, g1o_t)
            LoHi = colfilter(Lo, g1o_t)
            HiLo = colfilter(Hi, g0o_t)
            HiHi = colfilter(Hi, g1o_t)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            grad_yhr1 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi1 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhi1, None, None, None, None, None, None


class xfm1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Hi = rowfilter(input, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr1 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi1 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        
        Yl = LoLo
        return Yl, Yhr1, Yhi1

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm1no_l1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhi1, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        y = rowfilter(colfilter(yl, g0o), g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhi1 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            grad_yhr1 = None
            grad_yhi1 = None


            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhi1, None, None, None, None, None, None


class xfm1no_l1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Yhr1 = None
        Yhi1 = None

        
        Yl = LoLo
        return Yl, Yhr1, Yhi1

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm2(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhi1, yhi2, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhi1 = None
        grad_yhi2 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            Hi = rowfilter(grad_y, g1o_t)
            LoHi = colfilter(Lo, g1o_t)
            HiLo = colfilter(Hi, g0o_t)
            HiHi = colfilter(Hi, g1o_t)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            grad_yhr1 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi1 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhi1, grad_yhi2, None, None, None, None, None, None


class xfm2(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Hi = rowfilter(input, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr1 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi1 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm2no_l1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhi1, yhi2, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        y = rowfilter(colfilter(yl, g0o), g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhi1 = None
        grad_yhi2 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            grad_yhr1 = None
            grad_yhi1 = None


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhi1, grad_yhi2, None, None, None, None, None, None


class xfm2no_l1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Yhr1 = None
        Yhi1 = None

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm3(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhi1, yhi2, yhi3, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            Hi = rowfilter(grad_y, g1o_t)
            LoHi = colfilter(Lo, g1o_t)
            HiLo = colfilter(Hi, g0o_t)
            HiHi = colfilter(Hi, g1o_t)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            grad_yhr1 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi1 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhi1, grad_yhi2, grad_yhi3, None, None, None, None, None, None


class xfm3(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Hi = rowfilter(input, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr1 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi1 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm3no_l1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhi1, yhi2, yhi3, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        y = rowfilter(colfilter(yl, g0o), g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            grad_yhr1 = None
            grad_yhi1 = None


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhi1, grad_yhi2, grad_yhi3, None, None, None, None, None, None


class xfm3no_l1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Yhr1 = None
        Yhi1 = None

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm4(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhi1, yhi2, yhi3, yhi4, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            Hi = rowfilter(grad_y, g1o_t)
            LoHi = colfilter(Lo, g1o_t)
            HiLo = colfilter(Hi, g0o_t)
            HiHi = colfilter(Hi, g1o_t)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            grad_yhr1 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi1 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, None, None, None, None, None, None


class xfm4(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Hi = rowfilter(input, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr1 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi1 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm4no_l1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhi1, yhi2, yhi3, yhi4, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        y = rowfilter(colfilter(yl, g0o), g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            grad_yhr1 = None
            grad_yhi1 = None


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, None, None, None, None, None, None


class xfm4no_l1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Yhr1 = None
        Yhi1 = None

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm5(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhr5, yhi1, yhi2, yhi3, yhi4, yhi5, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr5[:,:,0:6:5], yhi5[:,:,0:6:5])
        hl = c2q(yhr5[:,:,2:4:1], yhi5[:,:,2:4:1])
        hh = c2q(yhr5[:,:,1:5:3], yhi5[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhr5 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None
        grad_yhi5 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            Hi = rowfilter(grad_y, g1o_t)
            LoHi = colfilter(Lo, g1o_t)
            HiLo = colfilter(Hi, g0o_t)
            HiHi = colfilter(Hi, g1o_t)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            grad_yhr1 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi1 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr5 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi5 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhr5, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, grad_yhi5, None, None, None, None, None, None


class xfm5(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Hi = rowfilter(input, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr1 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi1 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr5 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi5 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4, Yhr5, Yhi5

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4, grad_Yhr5, grad_Yhi5):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr5[:,:,0:6:5], grad_Yhi5[:,:,0:6:5])
            hl = c2q(grad_Yhr5[:,:,2:4:1], grad_Yhi5[:,:,2:4:1])
            hh = c2q(grad_Yhr5[:,:,1:5:3], grad_Yhi5[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm5no_l1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhr5, yhi1, yhi2, yhi3, yhi4, yhi5, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr5[:,:,0:6:5], yhi5[:,:,0:6:5])
        hl = c2q(yhr5[:,:,2:4:1], yhi5[:,:,2:4:1])
        hh = c2q(yhr5[:,:,1:5:3], yhi5[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        y = rowfilter(colfilter(yl, g0o), g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhr5 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None
        grad_yhi5 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            grad_yhr1 = None
            grad_yhi1 = None


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr5 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi5 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhr5, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, grad_yhi5, None, None, None, None, None, None


class xfm5no_l1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Yhr1 = None
        Yhi1 = None

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr5 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi5 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4, Yhr5, Yhi5

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4, grad_Yhr5, grad_Yhi5):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr5[:,:,0:6:5], grad_Yhi5[:,:,0:6:5])
            hl = c2q(grad_Yhr5[:,:,2:4:1], grad_Yhi5[:,:,2:4:1])
            hh = c2q(grad_Yhr5[:,:,1:5:3], grad_Yhi5[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm6(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhr5, yhr6, yhi1, yhi2, yhi3, yhi4, yhi5, yhi6, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr6[:,:,0:6:5], yhi6[:,:,0:6:5])
        hl = c2q(yhr6[:,:,2:4:1], yhi6[:,:,2:4:1])
        hh = c2q(yhr6[:,:,1:5:3], yhi6[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr5[:,:,0:6:5], yhi5[:,:,0:6:5])
        hl = c2q(yhr5[:,:,2:4:1], yhi5[:,:,2:4:1])
        hh = c2q(yhr5[:,:,1:5:3], yhi5[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhr5 = None
        grad_yhr6 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None
        grad_yhi5 = None
        grad_yhi6 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            Hi = rowfilter(grad_y, g1o_t)
            LoHi = colfilter(Lo, g1o_t)
            HiLo = colfilter(Hi, g0o_t)
            HiHi = colfilter(Hi, g1o_t)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            grad_yhr1 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi1 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr5 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi5 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr6 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi6 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhr5, grad_yhr6, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, grad_yhi5, grad_yhi6, None, None, None, None, None, None


class xfm6(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Hi = rowfilter(input, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr1 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi1 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr5 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi5 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr6 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi6 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4, Yhr5, Yhi5, Yhr6, Yhi6

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4, grad_Yhr5, grad_Yhi5, grad_Yhr6, grad_Yhi6):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr6[:,:,0:6:5], grad_Yhi6[:,:,0:6:5])
            hl = c2q(grad_Yhr6[:,:,2:4:1], grad_Yhi6[:,:,2:4:1])
            hh = c2q(grad_Yhr6[:,:,1:5:3], grad_Yhi6[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr5[:,:,0:6:5], grad_Yhi5[:,:,0:6:5])
            hl = c2q(grad_Yhr5[:,:,2:4:1], grad_Yhi5[:,:,2:4:1])
            hh = c2q(grad_Yhr5[:,:,1:5:3], grad_Yhi5[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm6no_l1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhr5, yhr6, yhi1, yhi2, yhi3, yhi4, yhi5, yhi6, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr6[:,:,0:6:5], yhi6[:,:,0:6:5])
        hl = c2q(yhr6[:,:,2:4:1], yhi6[:,:,2:4:1])
        hh = c2q(yhr6[:,:,1:5:3], yhi6[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr5[:,:,0:6:5], yhi5[:,:,0:6:5])
        hl = c2q(yhr5[:,:,2:4:1], yhi5[:,:,2:4:1])
        hh = c2q(yhr5[:,:,1:5:3], yhi5[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        y = rowfilter(colfilter(yl, g0o), g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhr5 = None
        grad_yhr6 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None
        grad_yhi5 = None
        grad_yhi6 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            grad_yhr1 = None
            grad_yhi1 = None


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr5 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi5 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr6 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi6 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhr5, grad_yhr6, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, grad_yhi5, grad_yhi6, None, None, None, None, None, None


class xfm6no_l1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Yhr1 = None
        Yhi1 = None

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr5 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi5 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr6 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi6 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4, Yhr5, Yhi5, Yhr6, Yhi6

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4, grad_Yhr5, grad_Yhi5, grad_Yhr6, grad_Yhi6):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr6[:,:,0:6:5], grad_Yhi6[:,:,0:6:5])
            hl = c2q(grad_Yhr6[:,:,2:4:1], grad_Yhi6[:,:,2:4:1])
            hh = c2q(grad_Yhr6[:,:,1:5:3], grad_Yhi6[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr5[:,:,0:6:5], grad_Yhi5[:,:,0:6:5])
            hl = c2q(grad_Yhr5[:,:,2:4:1], grad_Yhi5[:,:,2:4:1])
            hh = c2q(grad_Yhr5[:,:,1:5:3], grad_Yhi5[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm7(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhr5, yhr6, yhr7, yhi1, yhi2, yhi3, yhi4, yhi5, yhi6, yhi7, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr7[:,:,0:6:5], yhi7[:,:,0:6:5])
        hl = c2q(yhr7[:,:,2:4:1], yhi7[:,:,2:4:1])
        hh = c2q(yhr7[:,:,1:5:3], yhi7[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr6[:,:,0:6:5], yhi6[:,:,0:6:5])
        hl = c2q(yhr6[:,:,2:4:1], yhi6[:,:,2:4:1])
        hh = c2q(yhr6[:,:,1:5:3], yhi6[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr5[:,:,0:6:5], yhi5[:,:,0:6:5])
        hl = c2q(yhr5[:,:,2:4:1], yhi5[:,:,2:4:1])
        hh = c2q(yhr5[:,:,1:5:3], yhi5[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhr5 = None
        grad_yhr6 = None
        grad_yhr7 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None
        grad_yhi5 = None
        grad_yhi6 = None
        grad_yhi7 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            Hi = rowfilter(grad_y, g1o_t)
            LoHi = colfilter(Lo, g1o_t)
            HiLo = colfilter(Hi, g0o_t)
            HiHi = colfilter(Hi, g1o_t)
            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
            grad_yhr1 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi1 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr5 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi5 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr6 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi6 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr7 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi7 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhr5, grad_yhr6, grad_yhr7, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, grad_yhi5, grad_yhi6, grad_yhi7, None, None, None, None, None, None


class xfm7(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Hi = rowfilter(input, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)
        Yhr1 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi1 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr5 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi5 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr6 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi6 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr7 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi7 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4, Yhr5, Yhi5, Yhr6, Yhi6, Yhr7, Yhi7

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4, grad_Yhr5, grad_Yhi5, grad_Yhr6, grad_Yhi6, grad_Yhr7, grad_Yhi7):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr7[:,:,0:6:5], grad_Yhi7[:,:,0:6:5])
            hl = c2q(grad_Yhr7[:,:,2:4:1], grad_Yhi7[:,:,2:4:1])
            hh = c2q(grad_Yhr7[:,:,1:5:3], grad_Yhi7[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr6[:,:,0:6:5], grad_Yhi6[:,:,0:6:5])
            hl = c2q(grad_Yhr6[:,:,2:4:1], grad_Yhi6[:,:,2:4:1])
            hh = c2q(grad_Yhr6[:,:,1:5:3], grad_Yhi6[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr5[:,:,0:6:5], grad_Yhi5[:,:,0:6:5])
            hl = c2q(grad_Yhr5[:,:,2:4:1], grad_Yhi5[:,:,2:4:1])
            hh = c2q(grad_Yhr5[:,:,1:5:3], grad_Yhi5[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            

        return (grad_input,) + (None,) * 6


class ifm7no_l1(Function):

    @staticmethod
    def forward(ctx, yl, yhr1, yhr2, yhr3, yhr4, yhr5, yhr6, yhr7, yhi1, yhi2, yhi3, yhi4, yhi5, yhi6, yhi7, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        
        lh = c2q(yhr7[:,:,0:6:5], yhi7[:,:,0:6:5])
        hl = c2q(yhr7[:,:,2:4:1], yhi7[:,:,2:4:1])
        hh = c2q(yhr7[:,:,1:5:3], yhi7[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr6[:,:,0:6:5], yhi6[:,:,0:6:5])
        hl = c2q(yhr6[:,:,2:4:1], yhi6[:,:,2:4:1])
        hh = c2q(yhr6[:,:,1:5:3], yhi6[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr5[:,:,0:6:5], yhi5[:,:,0:6:5])
        hl = c2q(yhr5[:,:,2:4:1], yhi5[:,:,2:4:1])
        hh = c2q(yhr5[:,:,1:5:3], yhi5[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr4[:,:,0:6:5], yhi4[:,:,0:6:5])
        hl = c2q(yhr4[:,:,2:4:1], yhi4[:,:,2:4:1])
        hh = c2q(yhr4[:,:,1:5:3], yhi4[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr3[:,:,0:6:5], yhi3[:,:,0:6:5])
        hl = c2q(yhr3[:,:,2:4:1], yhi3[:,:,2:4:1])
        hh = c2q(yhr3[:,:,1:5:3], yhi3[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)

        lh = c2q(yhr2[:,:,0:6:5], yhi2[:,:,0:6:5])
        hl = c2q(yhr2[:,:,2:4:1], yhi2[:,:,2:4:1])
        hh = c2q(yhr2[:,:,1:5:3], yhi2[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)
        y = rowfilter(colfilter(yl, g0o), g0o)


        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        grad_yhr1 = None
        grad_yhr2 = None
        grad_yhr3 = None
        grad_yhr4 = None
        grad_yhr5 = None
        grad_yhr6 = None
        grad_yhr7 = None
        grad_yhi1 = None
        grad_yhi2 = None
        grad_yhi3 = None
        grad_yhi4 = None
        grad_yhi5 = None
        grad_yhi6 = None
        grad_yhi7 = None

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            grad_yhr1 = None
            grad_yhi1 = None


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr2 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi2 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr3 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi3 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr4 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi4 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr5 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi5 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr6 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi6 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr7 = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi7 = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

            grad_yl = LoLo

        return grad_yl, grad_yhr1, grad_yhr2, grad_yhr3, grad_yhr4, grad_yhr5, grad_yhr6, grad_yhr7, grad_yhi1, grad_yhi2, grad_yhi3, grad_yhi4, grad_yhi5, grad_yhi6, grad_yhi7, None, None, None, None, None, None


class xfm7no_l1(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        Yhr1 = None
        Yhi1 = None

        
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
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


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr3 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi3 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr4 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi4 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr5 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi5 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr6 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi6 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)


        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr7 = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi7 = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)

        Yl = LoLo
        return Yl, Yhr1, Yhi1, Yhr2, Yhi2, Yhr3, Yhi3, Yhr4, Yhi4, Yhr5, Yhi5, Yhr6, Yhi6, Yhr7, Yhi7

    @staticmethod
    def backward(ctx, grad_LoLo, grad_Yhr1, grad_Yhi1, grad_Yhr2, grad_Yhi2, grad_Yhr3, grad_Yhi3, grad_Yhr4, grad_Yhi4, grad_Yhr5, grad_Yhi5, grad_Yhr6, grad_Yhi6, grad_Yhr7, grad_Yhi7):
        h0o, h1o, h0a, h0b, h1a, h1b = ctx.saved_tensors
        grad_input = None
        # Use the special properties of the filters to get the time reverse
        h0o_t = h0o
        h1o_t = h1o
        h0a_t = h0b
        h0b_t = h0a
        h1a_t = h1b
        h1b_t = h1a

        if ctx.calc_gradients:
            ll = grad_LoLo
            
            lh = c2q(grad_Yhr7[:,:,0:6:5], grad_Yhi7[:,:,0:6:5])
            hl = c2q(grad_Yhr7[:,:,2:4:1], grad_Yhi7[:,:,2:4:1])
            hh = c2q(grad_Yhr7[:,:,1:5:3], grad_Yhi7[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr6[:,:,0:6:5], grad_Yhi6[:,:,0:6:5])
            hl = c2q(grad_Yhr6[:,:,2:4:1], grad_Yhi6[:,:,2:4:1])
            hh = c2q(grad_Yhr6[:,:,1:5:3], grad_Yhi6[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr5[:,:,0:6:5], grad_Yhi5[:,:,0:6:5])
            hl = c2q(grad_Yhr5[:,:,2:4:1], grad_Yhi5[:,:,2:4:1])
            hh = c2q(grad_Yhr5[:,:,1:5:3], grad_Yhi5[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr4[:,:,0:6:5], grad_Yhi4[:,:,0:6:5])
            hl = c2q(grad_Yhr4[:,:,2:4:1], grad_Yhi4[:,:,2:4:1])
            hh = c2q(grad_Yhr4[:,:,1:5:3], grad_Yhi4[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr3[:,:,0:6:5], grad_Yhi3[:,:,0:6:5])
            hl = c2q(grad_Yhr3[:,:,2:4:1], grad_Yhi3[:,:,2:4:1])
            hh = c2q(grad_Yhr3[:,:,1:5:3], grad_Yhi3[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)

            lh = c2q(grad_Yhr2[:,:,0:6:5], grad_Yhi2[:,:,0:6:5])
            hl = c2q(grad_Yhr2[:,:,2:4:1], grad_Yhi2[:,:,2:4:1])
            hh = c2q(grad_Yhr2[:,:,1:5:3], grad_Yhi2[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)
            grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)

            

        return (grad_input,) + (None,) * 6

