from __future__ import absolute_import

try:
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

import numpy as np
from collections import Iterable
from dtcwt_slim.utils import reflect
from dtcwt_slim.coeffs import biort as _biort


def as_column_vector(v):
    """Return *v* as a column vector with shape (N,1).

    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def _as_row_vector(v):
    """Return *v* as a row vector with shape (1, N).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v
    else:
        return v.T


def _as_row_tensor(h):
    if isinstance(h, torch.Tensor):
        h = torch.reshape(h, [1, -1])
    else:
        h = as_column_vector(h).T
        h = torch.tensor(h, dtype=torch.float32)
    return h


def _as_col_vector(v):
    """Return *v* as a column vector with shape (N,1).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def _as_col_tensor(h):
    if isinstance(h, torch.Tensor):
        h = torch.reshape(h, [-1, 1])
    else:
        h = as_column_vector(h)
        h = torch.tensor(h, dtype=torch.float32)
    return h


def _prepare_filter(h, ch=1, conv=True):
    # Check the shape of h is what we expect
    if len(h.shape) != 2:
        raise ValueError('Filter inputs must only have height and width ' +
                         'for conv_2d')

    # Have to reverse h as pytorch 2d conv is actually cross-correlation
    if conv:
        if h.shape[0] == 1:
            h = np.flip(h, 1)
        elif h.shape[1] == 1:
            h = np.flip(h, 0)
        else:
            raise ValueError('Conv 2d should be working on row/column vecs')

    h = np.reshape(h, [1, 1, *h.shape])
    h = np.repeat(h, repeats=ch, axis=0)
    h = np.copy(h)
    kernel = torch.tensor(h, dtype=torch.float32)
    return kernel


def _conv_2d(X, kernel, strides=[1,1], col=True):
    """
    Perform 2d convolution in pytorch. The kernel should be reversed before
    calling

    Data format must be NCHW. h should be a 2d numpy array.
    """

    # Check the shape of X is what we expect
    if len(X.shape) != 4:
        raise ValueError('X needs to be of shape [batch, ch, height, width] ' +
                         'for conv_2d')

    # convert the filter to one of size
    nchannels = X.shape[1]
    y = F.conv2d(X, kernel, stride=strides, groups=nchannels)

    return y


def _conv_2d_batch(X, kernel, strides=[1,1]):
    """
    Perform 2d convolution in pytorch.

    Data format must be NCHW. h should be a 2d numpy array. Instead of using
    grouped convolution, it puts the channels into the batch dimension
    """

    # Check the shape of X is what we expect
    if len(X.shape) != 4:
        raise ValueError('X needs to be of shape [batch, ch, height, width] ' +
                         'for conv_2d')

    # convert the filter to one of size
    # [size, size, nchannels, 1]
    batch, nchannels, iw, ih = X.shape
    X = torch.reshape(X, [batch*nchannels, 1, iw, ih])
    y = F.conv2d(X, kernel, stride=strides)

    # Reshape back to original layout
    ow, oh = y.shape[2:]
    y = torch.reshape(y, [batch, nchannels, ow, oh])

    return y


#  def _pad_rows(x, before, after):
    #  sx = x.shape[2]
    #  while before > 0 and after > 0:
        #  x = F.pad(x, (0, 0, before>0, after>0), mode='replicate')
        #  print(x)
        #  before = max(0, before-1)
        #  after = max(0, after-1)

        #  if before == 0 and after == 0:
            #  break
        #  x = F.pad(x, (0, 0, min(before, sx-1), min(after, sx-1)),
        #  mode='reflect')
        #  print(x)
        #  sx = x.shape[2]
    #  return x


def _pad(x, szs):
    """
    Pads an image by any amount (even larger than image size).

    Args:
        szs: tuple of pads for
        (xpad_before, xpad_after, ypad_before, ypad_after)

    pytorch can't handle padding by more than the dimension of the image.
    This wrapper allows us to build padding up successively.
    """
    sx, sy = x.shape[2:]
    if not isinstance(szs, Iterable):
        szs = [szs,] * 4
    szs = np.array(szs)

    gt = [szs[0] > sx-1, szs[1] > sx-1, szs[2] > sy-1, szs[3] > sy-1]
    while np.any(gt):
        # This creates an intermediate padding amount that will bring in
        # dimensions that are too big by the size of x.
        szs_step = np.int32(gt) * np.array([sx-1, sx-1, sy-1, sy-1])
        m = nn.ReflectionPad2d(tuple(szs_step))
        x = m(x)
        szs = szs - szs_step
        sx, sy = x.shape[2:]
        gt = [szs[0] > sx-1, szs[1] > sx-1, szs[2] > sy-1, szs[3] > sy-1]

    # Pad by the remaining amount
    m = nn.ReflectionPad2d(tuple(szs))
    x = m(x)
    return x


def prep_filt(h, c):
    h = _as_col_vector(h)[::-1]
    h = np.reshape(h, [1, 1, *h.shape])
    h = np.repeat(h, repeats=c, axis=0)
    h = np.copy(h)
    return torch.tensor(h, dtype=torch.float32)


class DTCWTFirstLayer(nn.Module):
    def __init__(self, in_channels, biort='near_sym_a'):
        super().__init__()
        self.biort = biort
        self.in_channels = in_channels

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        filts = _biort(biort)

        if len(filts) == 4:
            self.h0o = torch.nn.Parameter(prep_filt(filts[0], in_channels),
                                          requires_grad=False)
            self.g0o = torch.nn.Parameter(prep_filt(filts[1], in_channels),
                                          requires_grad=False)
            self.h1o = torch.nn.Parameter(prep_filt(filts[2], in_channels),
                                          requires_grad=False)
            self.g1o = torch.nn.Parameter(prep_filt(filts[3], in_channels),
                                          requires_grad=False)
            self.register_parameter('h2o', None)
            self.register_parameter('g2o', None)
        elif len(filts) == 6:
            self.h0o = torch.nn.Parameter(prep_filt(filts[0], in_channels),
                                          requires_grad=False)
            self.g0o = torch.nn.Parameter(prep_filt(filts[1], in_channels),
                                          requires_grad=False)
            self.h1o = torch.nn.Parameter(prep_filt(filts[2], in_channels),
                                          requires_grad=False)
            self.g1o = torch.nn.Parameter(prep_filt(filts[3], in_channels),
                                          requires_grad=False)
            self.h2o = torch.nn.Parameter(prep_filt(filts[4], in_channels),
                                          requires_grad=False)
            self.g2o = torch.nn.Parameter(prep_filt(filts[5], in_channels),
                                          requires_grad=False)

    def forward(self, input):
        xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)
        X = X[:,:,xe]
        #  if m % 2 == 0 and align:
            #  X = _pad(X, [m2 - 1, m2, 0, 0])
        #  else:
            #  X = _pad(X, [m2, m2, 0, 0])

        h_t = _prepare_filter(h, nchannels)
        Y = F.conv2d(X, h_t, stride=(1,1), groups=nchannels)

    def extra_repr(self):
        return


def colfilter(X, h):
    """
    Filter the cols of image *X* using filter vector *h*, without decimation.

    Parameters
    ----------
    X: torch.tensor
        A tensor of images whose rows are to be filtered. Needs to be of shape
        [batch, ch, h, w]
    h: torch.tensor
        The filter coefficients as a col tensor of shape (ch, 1, m, 1),
        where ch matches the ch dimension of X.


    Returns
    -------
    Y: torch.tensor
        the filtered image.

    Call prep_filt on a numpy array to ensure the ha and hb terms are of the
    correct form.

    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.
    If len(h) is even, each output sample is aligned with the mid point of each
    pair of input samples, and Y.shape = X.shape + [0 1].

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """
    m = h.shape[2]
    m2 = m // 2
    nchannels, r, c = X.shape[1:]

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the columns)
    xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)
    X = X[:,:,xe]
    Y = F.conv2d(X, h, stride=(1,1), groups=nchannels)

    return Y


def rowfilter(X, h):
    """
    Filter the rows of image *X* using filter vector *h*, without decimation.

    Parameters
    ----------
    X: torch.tensor
        A tensor of images whose rows are to be filtered. Needs to be of shape
        [batch, ch, h, w]
    h: torch.tensor
        The filter coefficients as a col tensor of shape (ch, 1, m, 1),
        where ch matches the ch dimension of X.

    Returns
    -------
    Y: torch.tensor
        the filtered image.

    Call prep_filt on a numpy array to ensure the ha and hb terms are of the
    correct form.

    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.
    If len(h) is even, each output sample is aligned with the mid point of each
    pair of input samples, and Y.shape = X.shape + [0 1].

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """
    # Do symmetric padding on the input
    m = h.shape[2]
    m2 = m // 2
    nchannels, r, c = X.shape[1:]
    xe = reflect(np.arange(-m2, c+m2, dtype=np.int), -0.5, c-0.5)
    X = X[:,:,:,xe]

    Y = F.conv2d(X, h.transpose(3,2), stride=(1,1), groups=nchannels)
    return Y


def coldfilt(X, ha, hb, hahb_pos=True):
    """
    Filter the columns of image X using the two filters ha and hb =
    reverse(ha).

    Parameters
    ----------
    X: torch.tensor
        The input, of size [batch, ch, h, w]
    ha: torch.tensor
        Filter to be used on the odd samples of x. Should be a a column tensor
        of shape (ch, 1, m, 1), where ch matches the ch dimension of X.
    hb: torch.tensor
        Filter to be used on the even samples of x. Should be a a column tensor
        of shape (ch, 1, m, 1), where ch matches the ch dimension of X.
    hahb_pos: bool
        True if np.sum(ha*hb) > 0

    Returns
    -------
    Y: torch.tensor
        Decimated result from convolving columns of X with ha and hb

    Call prep_filt on a numpy array to ensure the ha and hb terms are of the
    correct form.

    Both filters should be even length, and h should be approx linear
    phase with a quarter sample (i.e. an :math:`e^{j \pi/4}`) advance from
    its mid pt (i.e. :math:`|h(m/2)| > |h(m/2 + 1)|`)::

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    :raises ValueError if the number of rows in X is not a multiple of 4, the
        length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """
    ch, r, c = X.shape[1:]
    r2 = r // 2
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))
    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))
    m = ha.shape[2]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    rows = r2

    # Symmetrically extend with repeat of end samples.
    # Pad only the rows of X
    xe = reflect(np.arange(-m, r+m), -0.5, r-0.5)
    X = X[:,:,xe]

    # Take the odd and even rows of X
    X_odd = X[:,:, 2:r + 2 * m - 2:2, :]
    X_even = X[:,:, 3:r + 2 * m - 1:2, :]
    a_rows = F.conv2d(X_odd, ha, stride=(2,1), groups=ch)
    b_rows = F.conv2d(X_even, hb, stride=(2,1), groups=ch)

    # Stack a_rows and b_rows (both of shape [Batch, ch, r/4, c]) along the
    # third dimension to make a tensor of shape [Batch, ch, r/4, 2, c].
    if hahb_pos:
        Y = torch.stack((a_rows, b_rows), dim=-2)
    else:
        Y = torch.stack((b_rows, a_rows), dim=-2)

    # Reshape result to be shape [Batch, ch, r/2, c]. This reshaping interleaves
    # the columns
    Y = torch.reshape(Y, (-1, ch, rows, c))

    return Y


def rowdfilt(X, ha, hb, hahb_pos=True):
    """
    Filter the rows of image X using the two filters ha and hb =
    reverse(ha).

    Parameters
    ----------
    X: torch.tensor
        The input, of size [batch, ch, h, w]
    ha: torch.tensor
        Filter to be used on the odd samples of x. Should be a a column tensor
        of shape (ch, 1, m, 1), where ch matches the ch dimension of X.
    hb: torch.tensor
        Filter to be used on the even samples of x. Should be a a column tensor
        of shape (ch, 1, m, 1), where ch matches the ch dimension of X.
    hahb_pos: bool
        True if np.sum(ha*hb) > 0

    Returns
    -------
    Y: torch.tensor
        Decimated result from convolving columns of X with ha and hb

    Both filters should be even length, and h should be approx linear
    phase with a quarter sample (i.e. an :math:`e^{j \pi/4}`) advance from
    its mid pt (i.e. :math:`|h(m/2)| > |h(m/2 + 1)|`)::

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    :raises ValueError if the number of rows in X is not a multiple of 4, the
        length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """
    ch, r, c = X.shape[1:]
    c2 = c // 2
    if c % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4\n' +
                         'X was {}'.format(X.shape))

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))

    m = ha.shape[2]
    if m % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))

    # Do the 2d convolution, but only evaluated at every second sample
    # for both X_odd and X_even
    cols = c2

    # Symmetrically extend with repeat of end samples.
    # Pad only the second dimension of the tensor X (the rows).
    xe = reflect(np.arange(-m, c+m), -0.5, c-0.5)
    X = X[:,:,:,xe]

    # Take the odd and even columns of X
    X_odd = X[:,:,:,2:c + 2 * m - 2:2]
    X_even = X[:,:,:,3:c + 2 * m - 2:2]
    a_cols = F.conv2d(X_odd, ha.transpose(2,3), stride=(1,2), groups=ch)
    b_cols = F.conv2d(X_even, hb.transpose(2,3), stride=(1,2), groups=ch)

    # Stack a_cols and b_cols (both of shape [Batch, ch, r, c/4]) along the
    # fourth dimension to make a tensor of shape [Batch, ch, r, c/4, 2].
    if hahb_pos:
        Y = torch.stack((a_cols, b_cols), dim=-1)
    else:
        Y = torch.stack((b_cols, a_cols), dim=-1)

    # Reshape result to be shape [Batch, ch, r, c/2]. This reshaping interleaves
    # the columns
    Y = torch.reshape(Y, (-1, ch, r, cols))

    return Y


def colifilt(X, ha, hb, hahb_pos=True):
    """
    Filter the columns of image X using the two filters ha and hb =
    reverse(ha).

    Parameters
    ----------
    X: torch.tensor
        The input, of size [batch, ch, h, w]
    ha: torch.tensor
        Filter to be used on the odd samples of x. Should be a a column tensor
        of shape (ch, 1, m, 1), where ch matches the ch dimension of X.
    hb: torch.tensor
        Filter to be used on the even samples of x. Should be a a column tensor
        of shape (ch, 1, m, 1), where ch matches the ch dimension of X.
    hahb_pos: bool
        True if np.sum(ha*hb) > 0

    Returns
    -------
    Y: torch.tensor
        Bigger result from convolving columns of X with ha and hb. Will be of
        shape [batch, ch, 2*h, w]

    Both filters should be even length, and h should be approx linear
    phase with a quarter sample (i.e. an :math:`e^{j \pi/4}`) advance from
    its mid pt (i.e. :math:`|h(m/2)| > |h(m/2 + 1)|`)::

    The output is interpolated by two from the input sample rate and the
    results from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """

    ch, r, c = X.shape[1:]
    if r % 2 != 0:
        raise ValueError('No. of rows in X must be a multiple of 2.\n' +
                         'X was {}'.format(X.shape))

    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same.\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))

    m = ha.shape[0]
    m2 = m // 2
    if ha.shape[0] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even.\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))

    xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)
    X = X[:,:,xe]

    ha_odd = ha[:, :, ::2]
    ha_even = ha[:, :, 1::2]
    hb_even = hb[:, :, ::2]
    hb_odd = hb[:, :, 1::2]

    if m2 % 2 == 0:
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        if np.sum(ha*hb) > 0:
            X1, X2 = X[:,:, 1:r+m:2, :], X[:,:, 0:r+m-1:2, :]
        else:
            X1, X2 = X[:,:, 0:r+m-1:2, :], X[:,:, 1:r+m:2, :]

        # Stack along the filter dimension
        ha_t = torch.cat((ha_even_t, ha_odd_t), dim=-1)
        hb_t = torch.cat((hb_even_t, hb_odd_t), dim=-1)
        ya = _conv_2d(X1, hb_t)
        yb = _conv_2d(X2, ha_t)
        y1 = yb[:,::2,:-1]
        y2 = ya[:,::2,:-1]
        y3 = yb[:,1::2,1:]
        y4 = ya[:,1::2,1:]

    else:
        # m/2 is odd, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        if np.sum(ha*hb) > 0:
            X1, X2 = X[:,:, 2:r+m-1:2, :], X[:, :,1:r+m-2:2, :]
        else:
            X1, X2 = X[:,:, 1:r+m-2:2, :], X[:,:, 2:r+m-1:2, :]

        # Stack along the filter dimension
        ha_t = torch.cat((ha_odd_t, ha_even_t), dim=-1)
        hb_t = torch.cat((hb_odd_t, hb_even_t), dim=-1)

        # y1 has shape [batch, c*2 r2, c2]
        ya = _conv_2d(X2, ha_t)
        yb = _conv_2d(X1, hb_t)
        y1 = ya[:,::2]
        y2 = yb[:,::2]
        y3 = ya[:,1::2]
        y4 = yb[:,1::2]

    # Stack 4 tensors of shape [batch,ch, r2, c] into one tensor
    # [batch, ch, r2, 4, c]
    Y = torch.stack((y1, y2, y3, y4), dim=3)

    # Reshape to be [batch, ch,r * 2, c]. This interleaves the rows
    Y = torch.reshape(Y, (-1, ch, 2*r, c))

    return Y


def rowifilt(X, ha, hb):
    """
    Filter the row of image X using the two filters ha and hb =
    reverse(ha).

    Parameters
    ----------
    X: tf.Variable
        The input, of size [batch, ch, h, w]
    ha: np.array
        Filter to be used on the odd samples of x.
    hb: np.array
        Filter to bue used on the even samples of x.

    Returns
    -------
    Y: tf.Variable
        Bigger result from convolving columns of X with ha and hb. Will be of
        shape [batch, ch, 2*h, w]

    Both filters should be even length, and h should be approx linear
    phase with a quarter sample (i.e. an :math:`e^{j \pi/4}`) advance from
    its mid pt (i.e. :math:`|h(m/2)| > |h(m/2 + 1)|`)::

    The output is interpolated by two from the input sample rate and the
    results from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """
    ch, r, c = X.shape[1:]
    if c % 2 != 0:
        raise ValueError('No. of cols in X must be a multiple of 2.\n' +
                         'X was {}'.format(X.shape))

    ha = _as_row_vector(ha)
    hb = _as_row_vector(hb)
    if ha.shape != hb.shape:
        raise ValueError('Shapes of ha and hb must be the same.\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))

    m = ha.shape[1]
    m2 = m // 2
    if ha.shape[1] % 2 != 0:
        raise ValueError('Lengths of ha and hb must be even.\n' +
                         'ha was {}, hb was {}'.format(ha.shape, hb.shape))

    #  X = _pad(X, (0, 0, m2, m2))
    xe = reflect(np.arange(-m2, c+m2, dtype=np.int), -0.5, c-0.5)
    X = X[:,:,:,xe]

    ha_odd_t = _prepare_filter(ha[:,::2])
    ha_even_t = _prepare_filter(ha[:,1::2])
    hb_odd_t = _prepare_filter(hb[:,::2])
    hb_even_t = _prepare_filter(hb[:,1::2])

    if m2 % 2 == 0:
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.

        # Take the odd and even columns of X
        if np.sum(ha*hb) > 0:
            X1, X2 = X[:,:, :, 1:c+m:2], X[:,:, :, 0:c+m-1:2]
        else:
            X1, X2 = X[:,:, :, 0:c+m-1:2], X[:,:, :, 1:c+m:2]

        # Stack along the filter dimension
        ha_t = torch.cat((ha_even_t, ha_odd_t), dim=-1)
        hb_t = torch.cat((hb_even_t, hb_odd_t), dim=-1)
        ya = _conv_2d(X1, hb_t)
        yb = _conv_2d(X2, ha_t)
        y1 = yb[:,::2,:,:-1]
        y2 = ya[:,::2,:,:-1]
        y3 = yb[:,1::2,:,1:]
        y4 = ya[:,1::2,:,1:]

    else:
        # m/2 is odd, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.

        # Take the odd and even columns of X
        if np.sum(ha*hb) > 0:
            X1, X2 = X[:,:, :, 2:c+m-1:2], X[:,:, :, 1:c+m-2:2]
        else:
            X1, X2 = X[:,:, :, 1:c+m-2:2], X[:,:, :, 2:c+m-1:2]

        # Stack along the filter dimension
        ha_t = torch.cat((ha_odd_t, ha_even_t), dim=-1)
        hb_t = torch.cat((hb_odd_t, hb_even_t), dim=-1)

        # y1 has shape [batch, c*2 r2, c2]
        ya = _conv_2d(X2, ha_t)
        yb = _conv_2d(X1, hb_t)
        y1 = ya[:,::2]
        y2 = yb[:,::2]
        y3 = ya[:,1::2]
        y4 = yb[:,1::2]

    # Stack 4 tensors of shape [batch, ch, r, c2] into one tensor
    # [batch, ch, r, c2, 4]
    Y = torch.stack((y1, y2, y3, y4), dim=-1)

    # Reshape to be [batch, ch, r , c*2]. This interleaves the columns
    Y = torch.reshape(Y, (-1, ch, r, 2*c))

    return Y
