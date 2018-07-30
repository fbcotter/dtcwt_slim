from torch.autograd import Function

## Forward Templates

level1_fwd = """Lo = rowfilter(input, h0o)
        LoLo = colfilter(Lo, h0o)
        {hps}
        """

level1_nohps_fwd = """Yhr1 = None
        Yhi1 = None
"""

level1_hps_fwd = """Hi = rowfilter(input, h1o)
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
"""

level1_bwd = """
            {hps}
            """
level1_nohps_bwd = """grad_input = rowfilter(colfilter(ll, h0o_t), h0o_t)
"""
level1_hps_bwd = """lh = c2q(grad_Yhr1[:,:,0:6:5], grad_Yhi1[:,:,0:6:5])
            hl = c2q(grad_Yhr1[:,:,2:4:1], grad_Yhi1[:,:,2:4:1])
            hh = c2q(grad_Yhr1[:,:,1:5:3], grad_Yhi1[:,:,1:5:3])
            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            grad_input = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)
"""

level2plus_fwd = """
        Lo = rowdfilt(LoLo, h0b, h0a)
        Hi = rowdfilt(LoLo, h1b, h1a, highpass=True)
        LoLo = coldfilt(Lo, h0b, h0a)
        LoHi = coldfilt(Lo, h1b, h1a, highpass=True)
        HiLo = coldfilt(Hi, h0b, h0a)
        HiHi = coldfilt(Hi, h1b, h1a, highpass=True)

        deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
        deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
        deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

        Yhr{j} = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
        Yhi{j} = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)
"""

level2plus_bwd = """
            lh = c2q(grad_Yhr{j}[:,:,0:6:5], grad_Yhi{j}[:,:,0:6:5])
            hl = c2q(grad_Yhr{j}[:,:,2:4:1], grad_Yhi{j}[:,:,2:4:1])
            hh = c2q(grad_Yhr{j}[:,:,1:5:3], grad_Yhi{j}[:,:,1:5:3])
            Hi = colifilt(hh, h1b_t, h1a_t, True) + colifilt(hl, h0b_t, h0a_t)
            Lo = colifilt(lh, h1b_t, h1a_t, True) + colifilt(ll, h0b_t, h0a_t)
            ll = rowifilt(Hi, h1b_t, h1a_t, True) + rowifilt(Lo, h0b_t, h0a_t)"""

xfm = """
class xfm{J}{skip_hps}(Function):

    @staticmethod
    def forward(ctx, input, h0o, h1o, h0a, h0b, h1a, h1b):
        ctx.save_for_backward(h0o, h1o, h0a, h0b, h1a, h1b)
        ctx.calc_gradients = input.requires_grad
        batch, ch, r, c = input.shape
        {level1}{level2plus}
        Yl = LoLo
        return Yl, {fwd_out}

    @staticmethod
    def backward(ctx, grad_LoLo, {bwd_in}):
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
            {level2plusbwd}{level1bwd}

        return (grad_input,) + (None,) * 6

"""

## Inverse Templates

level1_fwd_inv = """
        {hps}"""
level1_nohps_fwd_inv = """y = rowfilter(colfilter(yl, g0o), g0o)
"""
level1_hps_fwd_inv = """lh = c2q(yhr1[:,:,0:6:5], yhi1[:,:,0:6:5])
        hl = c2q(yhr1[:,:,2:4:1], yhi1[:,:,2:4:1])
        hh = c2q(yhr1[:,:,1:5:3], yhi1[:,:,1:5:3])
        Hi = colfilter(hh, g1o) + colfilter(hl, g0o)
        Lo = colfilter(lh, g1o) + colfilter(ll, g0o)
        y = rowfilter(Hi, g1o) + rowfilter(Lo, g0o)
"""

level1_bwd_inv = """Lo = rowfilter(grad_y, g0o_t)
            LoLo = colfilter(Lo, g0o_t)
            {hps}
"""

level1_nohps_bwd_inv = """grad_yhr1 = None
            grad_yhi1 = None
"""

level1_hps_bwd_inv = """Hi = rowfilter(grad_y, g1o_t)
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
"""

level2plus_fwd_inv = """
        lh = c2q(yhr{j}[:,:,0:6:5], yhi{j}[:,:,0:6:5])
        hl = c2q(yhr{j}[:,:,2:4:1], yhi{j}[:,:,2:4:1])
        hh = c2q(yhr{j}[:,:,1:5:3], yhi{j}[:,:,1:5:3])
        Hi = colifilt(hh, g1b, g1a, True) + colifilt(hl, g0b, g0a)
        Lo = colifilt(lh, g1b, g1a, True) + colifilt(ll, g0b, g0a)
        ll = rowifilt(Hi, g1b, g1a, True) + rowifilt(Lo, g0b, g0a)"""

level2plus_bwd_inv = """
            Lo = rowdfilt(LoLo, g0b_t, g0a_t)
            Hi = rowdfilt(LoLo, g1b_t, g1a_t, highpass=True)
            LoLo = coldfilt(Lo, g0b_t, g0a_t)
            LoHi = coldfilt(Lo, g1b_t, g1a_t, highpass=True)
            HiLo = coldfilt(Hi, g0b_t, g0a_t)
            HiHi = coldfilt(Hi, g1b_t, g1a_t, highpass=True)

            deg15r, deg15i, deg165r, deg165i = q2c(LoHi)
            deg45r, deg45i, deg135r, deg135i = q2c(HiHi)
            deg75r, deg75i, deg105r, deg105i = q2c(HiLo)

            grad_yhr{j} = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=2)
            grad_yhi{j} = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=2)
"""

ifm = """
class ifm{J}{skip_hps}(Function):

    @staticmethod
    def forward(ctx, yl, {yhr_in}, {yhi_in}, g0o, g1o, g0a, g0b, g1a, g1b):
        ctx.save_for_backward(g0o, g1o, g0a, g0b, g1a, g1b)
        ctx.calc_gradients = yl.requires_grad
        batch, ch, r, c = yl.shape
        ll = yl
        {level2plus}{level1}

        return y

    @staticmethod
    def backward(ctx, grad_y):
        g0o, g1o, g0a, g0b, g1a, g1b = ctx.saved_tensors
        grad_yl = None
        {grad_yhr_init}
        {grad_yhi_init}

        # Use the special properties of the filters to get the time reverse
        g0o_t = g0o
        g1o_t = g1o
        g0a_t = g0b
        g0b_t = g0a
        g1a_t = g1b
        g1b_t = g1a

        if ctx.calc_gradients:
            {level1bwd}{level2plusbwd}
            grad_yl = LoLo

        return grad_yl, {grad_yhr_ret}, {grad_yhi_ret}, None, None, None, None, None, None

"""

#  FAST_FILTS = True
# Use the above templates to create all the code for DTCWT functions with layers
f = open('/scratch/fbc23/repos/fbcotter/dtcwt_slim/dtcwt_slim/torch/transform_funcs.py', 'w')
f.write("""'''This file was automatically generated by running transform_templates.py'''
""")
f.write('import torch\n')
f.write('from torch.autograd import Function\n')
#  f.write('from dtcwt_slim.torch.lowlevel import colfilter, rowfilter\n')
f.write('from dtcwt_slim.torch.lowlevel import ColFilter as colfilter, RowFilter as rowfilter\n')
f.write('from dtcwt_slim.torch.lowlevel import coldfilt, rowdfilt\n')
f.write('from dtcwt_slim.torch.lowlevel import colifilt, rowifilt, q2c, c2q\n')
for J in range(1,8):
    for skip_hps in (False, True):
        if skip_hps:
            suffix = 'no_l1'
        else:
            suffix = ''
        yhr_in = ', '.join(['yhr{}'.format(j) for j in range(1,J+1)])
        yhi_in = ', '.join(['yhi{}'.format(j) for j in range(1,J+1)])
        level2plus = "\n".join(
            [level2plus_fwd_inv.format(j=j) for j in range(J,1,-1)])
        level1 = level1_fwd_inv.format(hps=level1_nohps_fwd_inv if skip_hps else
                                       level1_hps_fwd_inv)
        grad_yhr_init = '\n        '.join(['grad_yhr{} = None'.format(j) for j in
                                   range(1,J+1)])
        grad_yhi_init = '\n        '.join(['grad_yhi{} = None'.format(j) for j in
                                   range(1,J+1)])
        level1bwd = level1_bwd_inv.format(hps=level1_nohps_bwd_inv if skip_hps
                                          else level1_hps_bwd_inv)
        level2plusbwd = "\n".join(
            [level2plus_bwd_inv.format(j=j) for j in range(2,J+1)])
        grad_yhr_ret = ", ".join(['grad_yhr{}'.format(j) for j in range(1,J+1)])
        grad_yhi_ret = ", ".join(['grad_yhi{}'.format(j) for j in range(1,J+1)])

        f.write(ifm.format(
            J=J,
            skip_hps=suffix,
            yhr_in=yhr_in,
            yhi_in=yhi_in,
            level2plus=level2plus,
            level1=level1,
            grad_yhr_init=grad_yhr_init,
            grad_yhi_init=grad_yhi_init,
            level1bwd=level1bwd,
            level2plusbwd=level2plusbwd,
            grad_yhr_ret=grad_yhr_ret,
            grad_yhi_ret=grad_yhi_ret,
        ))

        fwd_out = ", ".join(
            ['Yhr{j}, Yhi{j}'.format(j=j) for j in range(1,J+1)])
        bwd_in = ", ".join(
            ['grad_Yhr{j}, grad_Yhi{j}'.format(j=j) for j in range(1,J+1)])
        level2plus = '\n'.join(
            [level2plus_fwd.format(j=j) for j in range(2,J+1)])
        level2plusbwd = '\n'.join(
            [level2plus_bwd.format(j=j) for j in range(J,1,-1)])

        f.write(xfm.format(
            skip_hps=suffix,
            level1=level1_fwd.format(hps=level1_nohps_fwd if skip_hps else
                                     level1_hps_fwd),
            level2plus=level2plus,
            fwd_out=fwd_out,
            bwd_in=bwd_in,
            level1bwd=level1_bwd.format(hps=level1_nohps_bwd if skip_hps else
                                        level1_hps_bwd),
            level2plusbwd=level2plusbwd,
            J=J))

f.close()
