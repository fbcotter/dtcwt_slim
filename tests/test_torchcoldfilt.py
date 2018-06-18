from pytest import raises

import numpy as np
from dtcwt_slim.coeffs import qshift
from dtcwt.numpy.lowlevel import coldfilt as np_coldfilt

import tests.datasets as datasets
from dtcwt_slim.torch.lowlevel import coldfilt
from dtcwt_slim.tf.lowlevel import coldfilt as tf_coldfilt
import tensorflow as tf
import torch
import py3nvml


def setup():
    global barbara, barbara_t
    global bshape, bshape_half
    global ref_coldfilt
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[1] //= 2
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0)

    # Some useful functions
    ref_coldfilt = lambda x, ha, hb: np.stack(
        [np_coldfilt(s, ha, hb) for s in x], axis=0)


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_odd_filter():
    with raises(ValueError):
        coldfilt(barbara_t, (-1,2,-1), (-1,2,1))


def test_different_size():
    with raises(ValueError):
        coldfilt(barbara_t, (-0.5,-1,2,1,0.5), (-1,2,-1))


def test_bad_input_size():
    with raises(ValueError):
        coldfilt(barbara_t[:,:,:511,:], (-1,1), (1,-1))


def test_good_input_size():
    coldfilt(barbara_t[:,:,:,:511], (-1,1), (1,-1))


def test_good_input_size_non_orthogonal():
    coldfilt(barbara_t[:,:,:,:511], (1,1), (1,1))


def test_output_size():
    y_op = coldfilt(barbara_t, (-1,1), (1,-1))
    assert list(y_op.shape)[1:] == bshape_half


def test_equal_small_in():
    ha = qshift('qshift_b')[0]
    hb = qshift('qshift_b')[1]
    im = barbara[:,0:4,0:4]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, ha, hb)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift1():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    ref = ref_coldfilt(barbara, ha, hb)
    y = coldfilt(barbara_t, ha, hb)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift2():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im = barbara[:, :508, :502]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_coldfilt(im, ha, hb)
    y = coldfilt(im_t, ha, hb)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_gradients():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0)
    y_t = coldfilt(im_t, ha, hb)
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    dx = torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))

    im_t2 = tf.expand_dims(tf.constant(barbara, tf.float32), axis=0)
    y_t2 = tf_coldfilt(im_t2, ha, hb)
    with tf.Session() as sess:
        dx2 = tf.gradients(y_t2, im_t2, grad_ys=dy)
        sess.run(tf.global_variables_initializer())
        dx2 = sess.run(dx2[0])

    np.testing.assert_array_almost_equal(dx2, dx, decimal=4)



# vim:sw=4:sts=4:et
