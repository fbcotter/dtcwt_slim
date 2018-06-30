from pytest import raises

import numpy as np
from dtcwt_slim.coeffs import qshift
from dtcwt.numpy.lowlevel import coldfilt as np_coldfilt
from dtcwt_slim.torch.lowlevel import rowdfilt, prep_filt
from dtcwt_slim.tf.lowlevel import rowdfilt as tf_rowdfilt

import tests.datasets as datasets
import tensorflow as tf
import torch
import py3nvml


def setup():
    global barbara, barbara_t
    global bshape, bshape_half
    global ref_rowdfilt, ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[2] //= 2
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0)
    ch = barbara_t.shape[1]

    # Some useful functions
    ref_rowdfilt = lambda x, ha, hb: np.stack(
        [np_coldfilt(s.T, ha, hb).T for s in x], axis=0)


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_odd_filter():
    with raises(ValueError):
        ha = prep_filt((-1,2,-1), ch)
        hb = prep_filt((-1,2,1), ch)
        rowdfilt(barbara_t, ha, hb)


def test_different_size():
    with raises(ValueError):
        ha = prep_filt((-0.5,-1,2,0.5), ch)
        hb = prep_filt((-1,2,1), ch)
        rowdfilt(barbara_t, ha, hb)


def test_bad_input_size():
    with raises(ValueError):
        ha = prep_filt((-1, 1), ch)
        hb = prep_filt((1, -1), ch)
        rowdfilt(barbara_t[:,:,:,:511], ha, hb)


def test_good_input_size():
    ha = prep_filt((-1, 1), ch)
    hb = prep_filt((1, -1), ch)
    rowdfilt(barbara_t[:,:,:511,:], ha, hb)


def test_good_input_size_non_orthogonal():
    ha = prep_filt((1, 1), ch)
    hb = prep_filt((1, -1), ch)
    rowdfilt(barbara_t[:,:,:511,:], ha, hb)


def test_output_size():
    ha = prep_filt((-1, 1), ch)
    hb = prep_filt((1, -1), ch)
    y_op = rowdfilt(barbara_t, ha, hb)
    assert list(y_op.shape[1:]) == bshape_half


def test_equal_small_in():
    ha = qshift('qshift_b')[0]
    hb = qshift('qshift_b')[1]
    im = barbara[:,0:4,0:4]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_rowdfilt(im, ha, hb)
    y = rowdfilt(im_t, prep_filt(ha, ch), prep_filt(hb, ch))
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift1():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    ref = ref_rowdfilt(barbara, ha, hb)
    y = rowdfilt(barbara_t, prep_filt(ha, ch), prep_filt(hb, ch), np.sum(ha*hb)>0)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift2():
    ha = qshift('qshift_b')[0]
    hb = qshift('qshift_b')[1]
    im = barbara[:, :502, :508]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_rowdfilt(im, ha, hb)
    y = rowdfilt(im_t, prep_filt(ha, ch), prep_filt(hb, ch), np.sum(ha*hb)>0)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift3():
    hb = qshift('qshift_c')[0]
    ha = qshift('qshift_c')[1]
    im = barbara[:, :502, :508]
    im_t = torch.unsqueeze(torch.tensor(im, dtype=torch.float32), dim=0)
    ref = ref_rowdfilt(im, ha, hb)
    y = rowdfilt(im_t, prep_filt(ha, ch), prep_filt(hb, ch), np.sum(ha*hb)>0)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_gradients():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32,
                                        requires_grad=True), dim=0)
    y_t = rowdfilt(im_t, prep_filt(ha, ch), prep_filt(hb, ch), np.sum(ha*hb)>0)
    dy = np.random.randn(*tuple(y_t.shape)).astype('float32')
    dx = torch.autograd.grad(y_t, im_t, grad_outputs=torch.tensor(dy))

    im_t2 = tf.expand_dims(tf.constant(barbara, tf.float32), axis=0)
    y_t2 = tf_rowdfilt(im_t2, ha, hb)
    with tf.Session() as sess:
        dx2_t = tf.gradients(y_t2, im_t2, grad_ys=dy)
        sess.run(tf.global_variables_initializer())
        dx2 = sess.run(dx2_t[0])

    np.testing.assert_array_almost_equal(dx2, dx[0].numpy(), decimal=4)
# vim:sw=4:sts=4:et
