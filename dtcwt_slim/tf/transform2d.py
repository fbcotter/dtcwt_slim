from __future__ import absolute_import

import numpy as np
import logging

from six.moves import xrange

from dtcwt_slim.coeffs import biort as _biort, qshift as _qshift
from dtcwt_slim.defaults import DEFAULT_BIORT, DEFAULT_QSHIFT

from dtcwt_slim.tf.lowlevel import coldfilt, rowdfilt, rowfilter, colfilter
from dtcwt_slim.tf.lowlevel import colifilt, rowifilt, complex_stack
from dtcwt_slim.tf.common import ComplexTensor


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

    Parameters
    ----------
    biort: str or np.array
        The biorthogonal wavelet family to use. If a string, will use this to
        call dtcwt_slim.coeffs.biort. If an array, will use these as the values.
    qshift: str or np.array
        The quarter shift wavelet family to use. If a string, will use this to
        call dtcwt_slim.coeffs.biort. If an array, will use these as the values.
    complex: bool
        If true, the highpass outputs will be returned as tensorflow complex
        variables. These will have shape [batch, ch, 6, h, w]. If false, they
        will be returned as pairs of real and imaginary outputs. These will have
        shape [batch, ch, 12, h, w].

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

    .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
    """
    def __init__(self, biort=DEFAULT_BIORT, qshift=DEFAULT_QSHIFT,
                 complex=True, device='GPU'):
        try:
            self.biort = _biort(biort)
        except TypeError:
            self.biort = biort

        # Load quarter sample shift wavelets
        try:
            self.qshift = _qshift(qshift)
        except TypeError:
            self.qshift = qshift
        self.complex = complex
        self.device = device

    def forward(self, X, nlevels=3, include_scale=False):
        """ Perform a forward transform on an image with multiple channels.

        Will perform the DTCWT independently on each channel. Data format for
        the input must have the height and width as the last 2 dimensions.
        Data input must be a tensor, variable or placeholder.

        Parameters
        ----------
        X: tf.Variable or np.array
            Input image which you wish to transform. Can be 2, 3, or 4
            dimensions, but height and width must be the last 2. If np.array is
            given, it will be converted to an tensorflow variable.
        nlevels: int
            Number of levels of the dtcwt transform to calculate.
        include_scale: bool
            Whether or not to return the lowpass results at each scale of the
            transform, or only at the highest scale (as is custom for
            multiresolution analysis)

        Returns
        -------
            Yl: tf.Variable
                Lowpass output
            Yh: list(tf.Variable)
                Highpass outputs. Will be complex and have one more dimension
                than the input representing the 6 orientations of the wavelets.
                This extra dimension will be the third last dimension. The first
                entry in the list is the first scale.
            Yscale: list(tf.Variable)
                Only returns if include_scale was true. A list of lowpass
                outputs at each scale.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
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
            if X_shape[0] is None:
                X = tf.expand_dims(X, axis=1)
            else:
                X = tf.expand_dims(X, axis=0)

        s = X.get_shape().as_list()[2:]
        size = '{}x{}'.format(s[0], s[1])
        name = 'dtcwt_fwd_{}'.format(size)

        # Do the dtcwt, now with a 4 dimensional input
        with tf.variable_scope(name):
            Yl, Yh, Yscale = self._forward_ops(X, nlevels)

        # Reshape it all again to match the input
        if len(X_shape) == 2:
            Yl = tf.squeeze(Yl, axis=[0,1])
            if self.complex:
                Yh = [tf.squeeze(s, axis=[0,1]) for s in Yh]
            else:
                fn = lambda x: tf.squeeze(x, axis=[0,1])
                Yh = [s.apply_func(fn) for s in Yh]
            Yscale = [tf.squeeze(s, axis=[0,1]) for s in Yscale]
        elif len(X_shape) == 3:
            if X_shape[0] is None:
                squeeze_ax = 1
            else:
                squeeze_ax = 0
            Yl = tf.squeeze(Yl, axis=[squeeze_ax])
            if self.complex:
                Yh = [tf.squeeze(s, axis=[squeeze_ax]) for s in Yh]
            else:
                fn = lambda x: tf.squeeze(x, axis=[squeeze_ax])
                Yh = [s.apply_func(fn) for s in Yh]
            Yscale = [tf.squeeze(s, axis=[squeeze_ax]) for s in Yscale]

        if include_scale:
            return Yl, Yh, Yscale
        else:
            return Yl, Yh

    def inverse(self, Yl, Yh):
        """
        Perform an inverse transform on an image with multiple channels.

        Parameters
        ----------
        Yl: tf.Variable
            The lowpass coefficients. Can be 2, 3, or 4 dimensions
        Yh: list(tf.Variable)
            The complex high pass coefficients. Must be compatible with the
            lowpass coefficients. Should have one more dimension. E.g if Yl
            was of shape [batch, ch, h, w], then the Yh's should be each of
            shape [batch, ch, 6, h', w'] (with h' and w' being dependent on the
            scale).

        Returns
        -------
        X: tf.Variable
            An array , X, compatible with the reconstruction.

        The (*d*, *l*)-th element of *gain_mask* is gain for subband with
        direction *d* at level *l*. If gain_mask[d,l] == 0, no computation is
        performed for band (d,l). Default *gain_mask* is all ones. Note that
        both *d* and *l* are zero-indexed.

        .. codeauthor:: Fergal Cotter <fbc23@cam.ac.uk>, Feb 2018
        """
        J = len(Yh)
        inshape = Yl.get_shape().as_list()

        # Reshape the inputs to all be 4d inputs of shape (batch, c, h, w)
        if len(inshape) == 2:
            Yl = tf.reshape(Yl, [1,1, *Yl.get_shape().as_list()])
            if self.complex:
                Yh = [tf.reshape(s, [1,1, *s.get_shape().as_list()])
                      for s in Yh]
            else:
                fn = lambda x: tf.reshape(x, [1,1, *x.get_shape().as_list()])
                Yh = [s.apply_func(fn) for s in Yh]
        elif len(inshape) == 3:
            Yl = tf.reshape(Yl, [1, *Yl.get_shape().as_list()])
            if self.complex:
                Yh = [tf.reshape(s, [1, *s.get_shape().as_list()]) for s in Yh]
            else:
                fn = lambda x: tf.reshape(x, [1, *x.get_shape().as_list()])
                Yh = [s.apply_func(fn) for s in Yh]

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

        Parameters
        ----------
        X: tf.Variable
            4D real array of size [batch, ch, h, w]
        nlevels: int
            Number of levels of wavelet decomposition

        Returns
        -------
        Yl: tf.Variable
        Yh: list(tf.Variables)
        Yscale: list(tf.Variables)

        Note
        ----
        This is the lowlevel implementation. You should call the forward method
        which will call this.
        """
        device = self.device

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
            h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b, h2a, h2b, g2a, g2b = self.qshift
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
            with tf.name_scope('scale0'):
                with tf.name_scope('convs'):
                    # Do odd top-level filters on cols.
                    Lo = colfilter(X, h0o, device=device, name='l0_col_low')
                    Hi = colfilter(X, h1o, device=device, name='l0_col_high')
                    if len(self.biort) >= 6:
                        Ba = colfilter(X, h2o)

                    # Do odd top-level filters on rows.
                    LoLo = rowfilter(Lo, h0o, device=device, name='l0_LoLo')
                    LoLo_shape = LoLo.get_shape().as_list()[-2:]

                    # Horizontal wavelet pair (15 & 165 degrees)
                    horiz = rowfilter(Hi, h0o, device=device, name='l0_LoHi')

                    # Vertical wavelet pair (75 & 105 degrees)
                    vertic = rowfilter(Lo, h1o, device=device, name='l0_HiLo')

                    # Diagonal wavelet pair (45 & 135 degrees)
                    if len(self.biort) >= 6:
                        diag = rowfilter(Ba, h2o, device=device, name='l0_HiHi')
                    else:
                        diag = rowfilter(Hi, h1o, device=device, name='l0_HiHi')

                with tf.name_scope('packing'):
                    if self.complex:
                        deg15, deg165 = q2c(horiz)
                        deg45, deg135 = q2c(diag)
                        deg75, deg105 = q2c(vertic)
                        # Pack all 6 tensors into one
                        Yh[0] = complex_stack(
                            [deg15, deg45, deg75, deg105, deg135, deg165],
                            axis=2, name='Yh_0_stack')
                    else:
                        deg15r, deg15i, deg165r, deg165i = q2c(horiz, False)
                        deg45r, deg45i, deg135r, deg135i = q2c(diag, False)
                        deg75r, deg75i, deg105r, deg105i = q2c(vertic, False)
                        Yh[0] = ComplexTensor([
                            tf.stack([deg15r, deg45r, deg75r, deg105r, deg135r,
                                      deg165r], axis=2, name='Yh0_r_stack'),
                            tf.stack([deg15i, deg45i, deg75i, deg105i, deg135i,
                                      deg165i], axis=2, name='Yh0_i_stack')
                        ])

                    Yscale[0] = LoLo

        # ############################ Level 2+ ################################
        # Uses the qshift filters
        for level in xrange(1, nlevels):
            with tf.name_scope('scale{}'.format(level)):
                with tf.name_scope('padding'):
                    row_size, col_size = LoLo_shape[0], LoLo_shape[1]
                    # If the row count of LoLo is not divisible by 4 (it will be
                    # divisible by 2), add 2 extra rows to make it so
                    if row_size % 4 != 0:
                        LoLo = tf.pad(LoLo, [[0,0], [0, 0], [1, 1], [0, 0]],
                                      'SYMMETRIC')

                    # If the col count of LoLo is not divisible by 4 (it will be
                    # divisible by 2), add 2 extra cols to make it so
                    if col_size % 4 != 0:
                        LoLo = tf.pad(LoLo, [[0,0], [0, 0], [0, 0], [1, 1]],
                                      'SYMMETRIC')

                with tf.name_scope('convs'):
                    # Do even Qshift filters on cols.
                    Lo = coldfilt(LoLo, h0b, h0a, device=device, name='l%d_col_low' % level)
                    Hi = coldfilt(LoLo, h1b, h1a, device=device, name='l%d_col_hi' % level)
                    if len(self.qshift) >= 12:
                        Ba = coldfilt(LoLo, h2b, h2a)

                    # Do even Qshift filters on rows.
                    LoLo = rowdfilt(Lo, h0b, h0a, device=device, name='l%d_LoLo' % level)
                    LoLo_shape = LoLo.get_shape().as_list()[-2:]

                    # Horizontal wavelet pair (15 & 165 degrees)
                    horiz = rowdfilt(Hi, h0b, h0a, device=device, name='l%d_LoHi' % level)

                    # Vertical wavelet pair (75 & 105 degrees)
                    vertic = rowdfilt(Lo, h1b, h1a, device=device, name='l%d_HiLo' % level)

                    # Diagonal wavelet pair (45 & 135 degrees)
                    if len(self.qshift) >= 12:
                        diag = rowdfilt(Ba, h2b, h2a, device=device, name='l%d_HiHi' % level)
                    else:
                        diag = rowdfilt(Hi, h1b, h1a, device=device, name='l%d_HiHi' % level)

                with tf.name_scope('packing'):
                    if self.complex:
                        deg15, deg165 = q2c(horiz, True)
                        deg45, deg135 = q2c(diag, True)
                        deg75, deg105 = q2c(vertic, True)

                        # Pack all 6 tensors into one
                        Yh[level] = complex_stack(
                            [deg15, deg45, deg75, deg105, deg135, deg165],
                            axis=2, name='Yh_{}_stack'.format(level))
                    else:
                        deg15r, deg15i, deg165r, deg165i = q2c(horiz, False)
                        deg45r, deg45i, deg135r, deg135i = q2c(diag, False)
                        deg75r, deg75i, deg105r, deg105i = q2c(vertic, False)
                        Yh[level] = ComplexTensor([
                            tf.stack([deg15r, deg45r, deg75r, deg105r, deg135r,
                                      deg165r], axis=2,
                                     name='Yh{}_r_stack'.format(level)),
                            tf.stack([deg15i, deg45i, deg75i, deg105i, deg135i,
                                      deg165i], axis=2,
                                     name='Yh{}_i_stack'.format(level))
                        ])

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

    def _inverse_ops(self, Yl, Yh):
        """Perform an *n*-level dual-tree complex wavelet (DTCWT) 2D
        reconstruction.

        Parameters
        ----------
        Yl: The lowpass output from a forward transform. Should be a
            tensorflow variable.
        Yh: list(tf.Variable)
            The list of highpass outputs from a forward transform.
            Should be tensorflow variables.

        Returns
        -------
        X: tf.Variable

        Note
        ----
        This is the lowlevel implementation. You should call the inverse method
        which will call this.
        """
        a = len(Yh)  # No of levels.
        device = self.device

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
            if self.complex:
                lh = c2q(tf.real(Yh[level][:,:,0:6:5]),
                         tf.imag(Yh[level][:,:,0:6:5]))
                hl = c2q(tf.real(Yh[level][:,:,2:4:1]),
                         tf.imag(Yh[level][:,:,2:4:1]))
                hh = c2q(tf.real(Yh[level][:,:,1:5:3]),
                         tf.imag(Yh[level][:,:,1:5:3]))
            else:
                lh = c2q(Yh[level].real[:,:,0:6:5],
                         Yh[level].imag[:,:,0:6:5])
                hl = c2q(Yh[level].real[:,:,2:4:1],
                         Yh[level].imag[:,:,2:4:1])
                hh = c2q(Yh[level].real[:,:,1:5:3],
                         Yh[level].imag[:,:,1:5:3])

            # Do even Qshift filters on columns.
            y1 = colifilt(Z, g0b, g0a, device=device, name='l%d_ll_col_low' % level) + \
                colifilt(lh, g1b, g1a, device=device, name='l%d_lh_col_high' % level)

            if len(self.qshift) >= 12:
                y2 = colifilt(hl, g0b, g0a, device=device, name='l%d_hl_col_low' % level)
                y2bp = colifilt(hh, g2b, g2a, device=device, name='l%d_hh_col_bp' % level)

                # Do even Qshift filters on rows.
                Z = rowifilt(y1, g0b, g0a, device=device, name='l%d_ll_row_low' % level) + \
                    rowifilt(y2, g1b, g1a, device=device, name='l%d_hl_row_high' % level) + \
                    rowifilt(y2bp, g2b, g2a, device=device, name='l%d_hh_row_bp' % level)
            else:
                y2 = colifilt(hl, g0b, g0a, device=device, name='l%d_hl_col_low' % level) + \
                    colifilt(hh, g1b, g1a, device=device, name='l%d_hh_col_high' % level)

                # Do even Qshift filters on rows.
                Z = rowifilt(y1, g0b, g0a, device=device, name='l%d_ll_row_low' % level) + \
                    rowifilt(y2, g1b, g1a, device=device, name='l%d_hl_row_high' % level)

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
            if self.complex:
                lh = c2q(tf.real(Yh[0][:,:,0:6:5]),
                         tf.imag(Yh[0][:,:,0:6:5]))
                hl = c2q(tf.real(Yh[0][:,:,2:4:1]),
                         tf.imag(Yh[0][:,:,2:4:1]))
                hh = c2q(tf.real(Yh[0][:,:,1:5:3]),
                         tf.imag(Yh[0][:,:,1:5:3]))
            else:
                lh = c2q(Yh[0].real[:,:,0:6:5],
                         Yh[0].imag[:,:,0:6:5])
                hl = c2q(Yh[0].real[:,:,2:4:1],
                         Yh[0].imag[:,:,2:4:1])
                hh = c2q(Yh[0].real[:,:,1:5:3],
                         Yh[0].imag[:,:,1:5:3])

            # Do odd top-level filters on columns.
            y1 = colfilter(Z, g0o, device=device, name='l0_ll_col_low') + \
                colfilter(lh, g1o, device=device, name='l0_lh_col_high')

            if len(self.biort) >= 6:
                y2 = colfilter(hl, g0o, device=device, name='l0_hl_col_low')
                y2bp = colfilter(hh, g2o, device=device, name='l0_hh_col_bp')

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o, device=device, name='l0_ll_row_low') + \
                    rowfilter(y2, g1o, device=device, name='l0_hl_row_high') + \
                    rowfilter(y2bp, g2o, device=device, name='l0_hh_row_bp')
            else:
                y2 = colfilter(hl, g0o, device=device, name='l0_hl_col_low') + \
                    colfilter(hh, g1o, device=device, name='l0_hh_col_high')

                # Do odd top-level filters on rows.
                Z = rowfilter(y1, g0o, device=device, name='l0_ll_row_low') + \
                    rowfilter(y2, g1o, device=device, name='l0_hl_row_high')

        return Z


def q2c(y, complex=True):
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

    if complex:
        p = tf.complex(a, b)    # p = (a + jb) / sqrt(2)
        q = tf.complex(d, -c)   # q = (d - jc) / sqrt(2)

        # Form the 2 highpasses in z.
        return (p - q, p + q)
    else:
        return (a-d, b+c, a+d, b-c)


def c2q(w_r, w_i, complex=True):
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
    ch, _, r, c = w_r.get_shape().as_list()[1:]
    w_r = w_r/np.sqrt(2)
    w_i = w_i/np.sqrt(2)
    x1 = w_r[:,:,0] + w_r[:,:,1]
    x2 = w_i[:,:,0] + w_i[:,:,1]
    x3 = w_i[:,:,0] - w_i[:,:,1]
    x4 = -w_r[:,:,0] + w_r[:,:,1]

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
