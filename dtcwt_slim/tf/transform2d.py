from __future__ import absolute_import

import numpy as np
import logging

from six.moves import xrange

from dtcwt_slim.coeffs import biort as _biort, qshift as _qshift
from dtcwt_slim.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT

from dtcwt_slim.tf.lowlevel import coldfilt, rowdfilt, rowfilter, colfilter
from dtcwt_slim.tf.lowlevel import colifilt, rowifilt

_GAIN_FUNC = 'einsum'

try:
    import tensorflow as tf
    from tensorflow.python.framework import dtypes
    tf_dtypes = frozenset(
        [dtypes.float32, dtypes.float64, dtypes.int8, dtypes.int16,
         dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.qint8, dtypes.qint32,
         dtypes.quint8, dtypes.complex64, dtypes.complex128,
         dtypes.float32_ref, dtypes.float64_ref, dtypes.int8_ref,
         dtypes.int16_ref, dtypes.int32_ref, dtypes.int64_ref, dtypes.uint8_ref,
         dtypes.qint8_ref, dtypes.qint32_ref, dtypes.quint8_ref,
         dtypes.complex64_ref, dtypes.complex128_ref]
    )
except ImportError:
    # The lack of tensorflow will be caught by the low-level routines.
    pass

np_dtypes = frozenset(
    [np.dtype('float16'), np.dtype('float32'), np.dtype('float64'),
     np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
     np.dtype('int64'), np.dtype('uint8'), np.dtype('uint16'),
     np.dtype('uint32'), np.dtype('complex64'), np.dtype('complex128')]
)


class Transform2d(object):
    """
    An implementation of the 2D DT-CWT via Tensorflow.

    :param biort: The biorthogonal wavelet family to use.
    :param qshift: The quarter shift wavelet family to use.

    .. note::

        *biort* and *qshift* are the wavelets which parameterise the transform.
        If *biort* or *qshift* are strings, they are used as an argument to the
        :py:func:`dtcwt.coeffs.biort` or :py:func:`dtcwt.coeffs.qshift`
        functions.  Otherwise, they are interpreted as tuples of vectors giving
        filter coefficients. In the *biort* case, this should be (h0o, g0o, h1o,
        g1o). In the *qshift* case, this should be (h0a, h0b, g0a, g0b, h1a,
        h1b, g1a, g1b).

    .. note::

        Calling the methods in this class with different inputs will slightly
        vary the results. If you call the
        :py:meth:`~dtcwt.tf.Transform2d.forward` or
        :py:meth:`~dtcwt.tf.Transform2d.forward_channels` methods with a numpy
        array, they load this array into a :py:class:`tf.Variable` and create
        the graph. Subsequent calls to :py:attr:`dtcwt.tf.Pyramid.lowpass` or
        other attributes in the pyramid will create a session and evaluate these
        parameters.  If the above methods are called with a tensorflow variable
        or placeholder, these will be used to create the graph. As such, to
        evaluate the results, you will need to look at the
        :py:attr:`dtcwt.tf.Pyramid.lowpass_op` attribute (calling the `lowpass`
        attribute will try to evaluate the graph with no initialized variables
        and likely result in a runtime error).

        The behaviour is similar for the inverse methods, except these return an
        array, rather than a Pyramid style class. If a
        :py:class:`dtcwt.tf.Pyramid` was created by calling the forward methods
        with a numpy array, providing this pyramid to the inverse methods will
        return a numpy array. If however a :py:class:`dtcwt.tf.Pyramid` was
        created by calling the forward methods with a tensorflow variable, the
        result from calling the inverse methods will also be a tensorflow
        variable.

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
    .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
    """

    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT):
        try:
            self.biort = _biort(biort)
        except TypeError:
            self.biort = biort

        # Load quarter sample shift wavelets
        try:
            self.qshift = _qshift(qshift)
        except TypeError:
            self.qshift = qshift

    def forward(self, X, nlevels=3, include_scale=False):
        """ Perform a forward transform on an image with multiple channels.

        Will perform the DTCWT independently on each channel. Data format for
        the input must have the height and width as the last 2 dimensions.
        Data input must be a tensor, variable or placeholder.

        :param X: Input image which you wish to transform.
        :param int nlevels: Number of levels of the dtcwt transform to
            calculate.
        :param bool include_scale: Whether or not to return the lowpass results
            at each scale of the transform, or only at the highest scale (as is
            custom for multiresolution analysis)

        :returns: A tuple of Yl, Yh

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
        """
        if not (isinstance(X, tf.Tensor) or
                isinstance(X, tf.Variable)):
            X = tf.Variable(X, dtype=tf.float32)

        X_shape = X.get_shape().as_list()
        if X.dtype is not tf.float32:
            X = tf.cast(X, tf.float32)

        # Reshape the inputs to all be 4d inputs of shape (batch, ch, h, w)
        if len(X_shape) == 2:
            X = tf.reshape(X, [1,1, *X.get_shape().as_list()])
        elif len(X_shape) == 3:
            X = tf.reshape(X, [1, *X.get_shape().as_list()])

        s = X.get_shape().as_list()[2:]
        size = '{}x{}'.format(s[0], s[1])
        name = 'dtcwt_fwd_{}'.format(size)

        # Do the dtcwt, now with a 4 dimensional input
        with tf.variable_scope(name):
            Yl, Yh, Yscale = self._forward_ops(X, nlevels)

        # Reshape it all again to match the input
        if len(X_shape) == 2:
            Yl = tf.squeeze(Yl, axis=[0,1])
            Yh = [tf.squeeze(s, axis=[0,1]) for s in Yh]
            Yscale = [tf.squeeze(s, axis=[0,1]) for s in Yscale]
        elif len(X_shape) == 3:
            Yl = tf.squeeze(Yl, axis=[0])
            Yh = [tf.squeeze(s, axis=[0]) for s in Yh]
            Yscale = [tf.squeeze(s, axis=[0]) for s in Yscale]

        if include_scale:
            return Yl, Yh, Yscale
        else:
            return Yl, Yh

    def inverse(self, Yl, Yh):
        """
        Perform an inverse transform on an image with multiple channels.

        Must provide with a tensorflow variable or placeholder (unlike the more
        general :py:meth:`~dtcwt.tf.Transform2d.inverse`).

        This is designed to work after calling the
        :py:meth:`~dtcwt.tf.Transform2d.forward_channels` method. You must use
        the same data_format for the inverse_channels as the one used for the
        forward_channels (unless you have explicitly reshaped the output).

        :param pyramid: A :py:class:`dtcwt.tf.Pyramid` like class holding
            the transform domain representation to invert

        :returns: An array , X, compatible with the reconstruction. Will be a tf
            Variable if the Pyramid was made with tf inputs, otherwise a numpy
            array.


        The (*d*, *l*)-th element of *gain_mask* is gain for subband with
        direction *d* at level *l*. If gain_mask[d,l] == 0, no computation is
        performed for band (d,l). Default *gain_mask* is all ones. Note that
        both *d* and *l* are zero-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, Sept 2001
        .. codeauthor:: Cian Shaffrey, Cambridge University, Sept 2001
        """
        J = len(Yh)
        inshape = Yl.get_shape().as_list()

        # Reshape the inputs to all be 3d inputs of shape (batch, h, w)
        if len(inshape) == 2:
            Yl = tf.reshape(Yl, [1,1, *Yl.get_shape().as_list()])
            Yh = [tf.reshape(s, [1,1, *s.get_shape().as_list()]) for s in Yh]
        elif len(inshape) == 3:
            Yl = tf.reshape(Yl, [1, *Yl.get_shape().as_list()])
            Yh = [tf.reshape(s, [1, *s.get_shape().as_list()]) for s in Yh]

        s = Yl.get_shape().as_list()[2:]
        size = '{}x{}'.format(s[0], s[1])
        name = 'dtcwt_inv_{}_{}scales'.format(size,J)

        # Do the dtcwt, now with a 4 dimensional input
        with tf.variable_scope(name):
            X = self._inverse_ops(Yl, Yh)

        # Reshape it all again to match the input
        if len(inshape) == 2:
            X = tf.squeeze(X, axis=[0,1])
        elif len(inshape) == 3:
            X = tf.squeeze(X, axis=[0])

        return X

    def _forward_ops(self, X, nlevels=3):
        """ Perform a *n*-level DTCWT-2D decompostion on a 2D matrix *X*.

        :param X: 3D real array of size [batch, h, w]
        :param nlevels: Number of levels of wavelet decomposition
        :param extended: True if a singleton dimension was added at the
            beginning of the input. Signal to remove afterwards.

        :returns: A tuple of Yl, Yh, Yscale
        """

        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b = self.qshift[:10]
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        # Check the shape and form of the input
        if X.dtype not in tf_dtypes:
            raise ValueError('X needs to be a tf variable or placeholder')

        X_size = X.get_shape().as_list()

        if len(X_size) != 4:
            raise ValueError(
                """The entered variable has too many dimensions {}. If
                the final dimension are colour channels, please enter each
                channel separately.""".format(X_size))

        # ############################ Resize #################################
        # The next few lines of code check to see if the image is odd in size,
        # if so an extra ... row/column will be added to the bottom/right of the
        # image
        initial_row_extend = 0
        initial_col_extend = 0
        # If the row count of X is not divisible by 2 then we need to
        # extend X by adding a row at the bottom
        if X_size[-2] % 2 != 0:
            bottom_row = X[:,:,-1:,:]
            X = tf.concat([X, bottom_row], axis=-2)
            initial_row_extend = 1

        # If the col count of X is not divisible by 2 then we need to
        # extend X by adding a col to the right
        if X_size[-1] % 2 != 0:
            right_col = X[:,:,:,-1:]
            X = tf.concat([X, right_col], axis=-1)
            initial_col_extend = 1

        extended_size = X.get_shape().as_list()

        if nlevels == 0:
            return X, (), ()

        # ########################### Initialise ###############################
        Yh = [None, ] * nlevels
        # This is only required if the user specifies a third output
        # component.
        Yscale = [None, ] * nlevels

        # ############################ Level 1 #################################
        # Uses the biorthogonal filters
        if nlevels >= 1:
            # Do odd top-level filters on cols.
            Lo = colfilter(X, h0o, name='l0_col_low')
            Hi = colfilter(X, h1o, name='l0_col_high')
            if len(self.biort) >= 6:
                Ba = colfilter(X, h2o)

            # Do odd top-level filters on rows.
            LoLo = rowfilter(Lo, h0o, name='l0_LoLo')
            LoLo_shape = LoLo.get_shape().as_list()[-2:]

            # Horizontal wavelet pair (15 & 165 degrees)
            horiz = q2c(rowfilter(Hi, h0o, name='l0_LoHi'))

            # Vertical wavelet pair (75 & 105 degrees)
            vertic = q2c(rowfilter(Lo, h1o, name='l0_HiLo'))

            # Diagonal wavelet pair (45 & 135 degrees)
            if len(self.biort) >= 6:
                diag = q2c(rowfilter(Ba, h2o, name='l0_HiHi'))
            else:
                diag = q2c(rowfilter(Hi, h1o, name='l0_HiHi'))

            # Pack all 6 tensors into one
            Yh[0] = tf.stack(
                [horiz[0], diag[0], vertic[0], vertic[1], diag[1], horiz[1]],
                axis=2)

            Yscale[0] = LoLo

        # ############################ Level 2+ ################################
        # Uses the qshift filters
        for level in xrange(1, nlevels):
            row_size, col_size = LoLo_shape[0], LoLo_shape[1]
            # If the row count of LoLo is not divisible by 4 (it will be
            # divisible by 2), add 2 extra rows to make it so
            if row_size % 4 != 0:
                LoLo = tf.pad(LoLo, [[0,0], [0, 0], [1, 1], [0, 0]], 'SYMMETRIC')

            # If the col count of LoLo is not divisible by 4 (it will be
            # divisible by 2), add 2 extra cols to make it so
            if col_size % 4 != 0:
                LoLo = tf.pad(LoLo, [[0,0], [0, 0], [0, 0], [1, 1]], 'SYMMETRIC')

            # Do even Qshift filters on cols.
            Lo = coldfilt(LoLo, h0b, h0a, name='l%d_col_low' % level)
            Hi = coldfilt(LoLo, h1b, h1a, name='l%d_col_hi' % level)
            if len(self.qshift) >= 12:
                Ba = coldfilt(LoLo, h2b, h2a)

            # Do even Qshift filters on rows.
            LoLo = rowdfilt(Lo, h0b, h0a, name='l%d_LoLo' % level)
            LoLo_shape = LoLo.get_shape().as_list()[-2:]

            # Horizontal wavelet pair (15 & 165 degrees)
            horiz = q2c(rowdfilt(Hi, h0b, h0a, name='l%d_LoHi' % level))

            # Vertical wavelet pair (75 & 105 degrees)
            vertic = q2c(rowdfilt(Lo, h1b, h1a, name='l%d_HiLo' % level))

            # Diagonal wavelet pair (45 & 135 degrees)
            if len(self.qshift) >= 12:
                diag = q2c(rowdfilt(Ba, h2b, h2a, name='l%d_HiHi' % level))
            else:
                diag = q2c(rowdfilt(Hi, h1b, h1a, name='l%d_HiHi' % level))

            # Pack all 6 tensors into one
            Yh[level] = tf.stack(
                [horiz[0], diag[0], vertic[0], vertic[1], diag[1], horiz[1]],
                axis=2)

            Yscale[level] = LoLo

        Yl = LoLo

        if initial_row_extend == 1 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in X_size))))
            logging.warn(
                """The bottom row and rightmost column have been duplicated,
                prior to decomposition.""")

        if initial_row_extend == 1 and initial_col_extend == 0:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in X_size))))
            logging.warn(
                'The bottom row has been duplicated, prior to decomposition.')

        if initial_row_extend == 0 and initial_col_extend == 1:
            logging.warn('The image entered is now a {0} NOT a {1}.'.format(
                'x'.join(list(str(s) for s in extended_size)),
                'x'.join(list(str(s) for s in X_size))))
            logging.warn(
                """The rightmost column has been duplicated, prior to
                decomposition.""")

        return Yl, Yh, Yscale

    def _inverse_ops(self, Yl, Yh, gain_mask=None):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 2D
        reconstruction.

        :param Yl: The lowpass output from a forward transform. Should be a
            tensorflow variable.
        :param Yh: The tuple of highpass outputs from a forward transform.
            Should be tensorflow variables.
        :param gain_mask: Gain to be applied to each subband.

        :returns: A tf.Variable holding the output

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with
        direction *d* at level *l*. If gain_mask[d,l] == 0, no computation is
        performed for band (d,l). Default *gain_mask* is all ones. Note that
        both *d* and *l* are zero-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2017
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, May 2002
        .. codeauthor:: Cian Shaffrey, Cambridge University, May 2002

        """
        a = len(Yh)  # No of levels.

        if gain_mask is None:
            gain_mask = np.ones((6, a))  # Default gain_mask.

        gain_mask = np.array(gain_mask)

        # If biort has 6 elements instead of 4, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.biort) == 4:
            h0o, g0o, h1o, g1o = self.biort
        elif len(self.biort) == 6:
            h0o, g0o, h1o, g1o, h2o, g2o = self.biort
        else:
            raise ValueError('Biort wavelet must have 6 or 4 components.')

        # If qshift has 12 elements instead of 8, then it's a modified
        # rotationally symmetric wavelet
        # FIXME: there's probably a nicer way to do this
        if len(self.qshift) == 8:
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = self.qshift
        elif len(self.qshift) == 12:
            h0a, h0b, g0a, g0b, h1a, h1b, \
                g1a, g1b, h2a, h2b, g2a, g2b = self.qshift
        else:
            raise ValueError('Qshift wavelet must have 12 or 8 components.')

        level = a - 1
        Z = Yl

        # This ensures that for level 1 we never do the following
        while level >= 1:
            lh = c2q(Yh[level][:,:,0:6:5], gain_mask[[0, 5], level])
            hl = c2q(Yh[level][:,:,2:4:1], gain_mask[[2, 3], level])
            hh = c2q(Yh[level][:,:,1:5:3], gain_mask[[1, 4], level])

            # Do even Qshift filters on columns.
            y1 = colifilt(Z, g0b, g0a, name='l%d_ll_col_low' % level) + \
                colifilt(lh, g1b, g1a, name='l%d_lh_col_high' % level)

            if len(self.qshift) >= 12:
                y2 = colifilt(hl, g0b, g0a, name='l%d_hl_col_low' % level)
                y2bp = colifilt(hh, g2b, g2a, name='l%d_hh_col_bp' % level)

                # Do even Qshift filters on rows.
                Z = rowifilt(y1, g0b, g0a, name='l%d_ll_row_low' % level) + \
                    rowifilt(y2, g1b, g1a, name='l%d_hl_row_high' % level) + \
                    rowifilt(y2bp, g2b, g2a, name='l%d_hh_row_bp' % level)
            else:
                y2 = colifilt(hl, g0b, g0a, name='l%d_hl_col_low' % level) + \
                    colifilt(hh, g1b, g1a, name='l%d_hh_col_high' % level)

                # Do even Qshift filters on rows.
                Z = rowifilt(y1, g0b, g0a, name='l%d_ll_row_low' % level) + \
                    rowifilt(y2, g1b, g1a, name='l%d_hl_row_high' % level)

            # Check size of Z and crop as required
            Z_r, Z_c = Z.get_shape().as_list()[-2:]
            S_r, S_c = Yh[level-1].get_shape().as_list()[-2:]
            # check to see if this result needs to be cropped for the rows
            if Z_r != S_r * 2:
                Z = Z[:,:, 1:-1, :]
            # check to see if this result needs to be cropped for the cols
            if Z_c != S_c * 2:
                Z = Z[:,:, :, 1:-1]

            # Assert that the size matches at this stage
            Z_r, Z_c = Z.get_shape().as_list()[-2:]
            if Z_r != S_r * 2 or Z_c != S_c * 2:
                raise ValueError(
                    'Sizes of highpasses {}x{} are not '.format(Z_r, Z_c) +
                    'compatible with {}x{} from next level'.format(S_r, S_c))

            level = level - 1

        if level == 0:
            lh = c2q(Yh[level][:,:,0:6:5], gain_mask[[0, 5], level])
            hl = c2q(Yh[level][:,:,2:4:1], gain_mask[[2, 3], level])
            hh = c2q(Yh[level][:,:,1:5:3], gain_mask[[1, 4], level])

            # Do odd top-level filters on columns.
            y1 = colfilter(Z, g0o, name='l0_ll_col_low') + \
                colfilter(lh, g1o, name='l0_lh_col_high')

            if len(self.biort) >= 6:
                y2 = colfilter(hl, g0o, name='l0_hl_col_low')
                y2bp = colfilter(hh, g2o, name='l0_hh_col_bp')

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o, name='l0_ll_row_low') + \
                    rowfilter(y2, g1o, name='l0_hl_row_high') + \
                    rowfilter(y2bp, g2o, name='l0_hh_row_bp')
            else:
                y2 = colfilter(hl, g0o, name='l0_hl_col_low') + \
                    colfilter(hh, g1o, name='l0_hh_col_high')

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o, name='l0_ll_row_low') + \
                    rowfilter(y2, g1o, name='l0_hl_row_high')

        return Z


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
    a, b = y[:,:, 0::2, 0::2], y[:,:, 0::2, 1::2]
    c, d = y[:,:, 1::2, 0::2], y[:,:, 1::2, 1::2]

    p = tf.complex(a / np.sqrt(2), b / np.sqrt(2))    # p = (a + jb) / sqrt(2)
    q = tf.complex(d / np.sqrt(2), -c / np.sqrt(2))   # q = (d - jc) / sqrt(2)

    # Form the 2 highpasses in z.
    return (p - q, p + q)


def c2q(w, gain):
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
    ch, _, r, c = w.get_shape().as_list()[1:]

    sc = np.sqrt(0.5) * gain
    P = w[:, :, 0] * sc[0] + w[:, :, 1] * sc[1]
    Q = w[:, :, 0] * sc[0] - w[:, :, 1] * sc[1]

    # Recover each of the 4 corners of the quads.
    x1 = tf.real(P)
    x2 = tf.imag(P)
    x3 = tf.imag(Q)
    x4 = -tf.real(Q)

    # Stack 2 inputs of shape [batch, ch, r, c] to [batch, ch, r, 2, c]
    x_rows1 = tf.stack([x1, x3], axis=-2)
    # Reshaping interleaves the results
    x_rows1 = tf.reshape(x_rows1, [-1, ch, 2 * r, c])
    # Do the same for the even columns
    x_rows2 = tf.stack([x2, x4], axis=-2)
    x_rows2 = tf.reshape(x_rows2, [-1, ch, 2 * r, c])

    # Stack the two [batch, ch, 2*r, c] tensors to [batch, ch, 2*r, c, 2]
    x_cols = tf.stack([x_rows1, x_rows2], axis=-1)
    y = tf.reshape(x_cols, [-1, ch, 2 * r, 2 * c])

    return y
