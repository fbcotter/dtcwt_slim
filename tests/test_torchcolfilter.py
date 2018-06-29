import numpy as np
from dtcwt_slim.coeffs import biort, qshift
from dtcwt.numpy.lowlevel import colfilter as np_colfilter

import tests.datasets as datasets
from dtcwt_slim.torch.lowlevel import colfilter, prep_filt
from dtcwt_slim.tf.lowlevel import colfilter as tf_colfilter
import tensorflow as tf
import torch
import py3nvml


def setup():
    global barbara, barbara_t, tf
    global bshape, bshape_extrarow
    global ref_colfilter, ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_extrarow = bshape[:]
    bshape_extrarow[1] += 1
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0)
    ch = barbara_t.shape[1]

    # Some useful functions
    ref_colfilter = lambda x, h: np.stack(
        [np_colfilter(s, h) for s in x], axis=0)


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_odd_size():
    h = [-1, 2, -1]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape


def test_even_size():
    h = [-1, -1]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape_extrarow


def test_qshift():
    h = qshift('qshift_a')[0]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape_extrarow


def test_biort():
    h = biort('antonini')[0]
    y_op = colfilter(barbara_t, prep_filt(h, ch))
    assert list(y_op.shape)[1:] == bshape


def test_even_size_batch():
    zero_t = torch.zeros((1, *barbara.shape), dtype=torch.float32)
    y = colfilter(zero_t, prep_filt([-1,1], ch))
    assert list(y.shape)[1:] == bshape_extrarow
    assert not np.any(y[:] != 0.0)


def test_equal_small_in():
    h = qshift('qshift_b')[0]
    im = barbara[:,0:4,0:4]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, 3))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_biort1():
    h = biort('near_sym_b')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_biort2():
    h = biort('near_sym_b')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift1():
    h = qshift('qshift_c')[0]
    ref = ref_colfilter(barbara, h)
    y = colfilter(barbara_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift2():
    h = qshift('qshift_c')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_colfilter(im, h)
    y = colfilter(im_t, prep_filt(h, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_gradients():
    h = biort('near_sym_b')[0]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0)
    y_t = colfilter(im_t, prep_filt(h, 3))
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    dx = torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))

    im_t2 = tf.expand_dims(tf.constant(barbara, tf.float32), axis=0)
    y_t2 = tf_colfilter(im_t2, h)
    with tf.Session() as sess:
        dx2_t = tf.gradients(y_t2, im_t2, grad_ys=dy)
        sess.run(tf.global_variables_initializer())
        dx2 = sess.run(dx2_t[0])

    np.testing.assert_array_almost_equal(dx2, dx[0].numpy(), decimal=4)
# vim:sw=4:sts=4:et
