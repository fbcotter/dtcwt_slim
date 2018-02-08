from pytest import raises

import numpy as np
from dtcwt_slim.coeffs import qshift
from dtcwt_slim.numpy.lowlevel import colifilt as np_colifilt
from dtcwt_slim.tf.lowlevel import rowifilt

import tests.datasets as datasets
import tensorflow as tf
import py3nvml


def setup():
    global barbara, barbara_t, tf
    global bshape, bshape_double
    global ref_rowifilt, t_shape, config
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_double = bshape[:]
    bshape_double[2] *= 2
    barbara_t = tf.expand_dims(tf.constant(barbara, dtype=tf.float32),
                               axis=0)

    # Some useful functions
    ref_rowifilt = lambda x, ha, hb: np.stack(
        [np_colifilt(s.T, ha, hb).T for s in x], axis=0)
    t_shape = lambda x: x.get_shape().as_list()


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert barbara_t.get_shape() == (1, 3, 512, 512)


def test_odd_filter():
    with raises(ValueError):
        rowifilt(barbara_t, (-1,2,-1), (-1,2,1))


def test_different_size_h():
    with raises(ValueError):
        rowifilt(barbara_t, (-1,2,1), (-0.5,-1,2,-1,0.5))


def test_zero_input():
    Y = rowifilt(barbara_t, (-1,1), (1,-1))
    with tf.Session(config=config) as sess:
        y = sess.run(Y, {barbara_t: [np.zeros_like(barbara)]})[0]
    assert np.all(y[:0] == 0)


def test_bad_input_size():
    with raises(ValueError):
        rowifilt(barbara_t[:,:, :,:511], (-1,1), (1,-1))


def test_good_input_size():
    rowifilt(barbara_t[:,:,:511,:], (-1,1), (1,-1))


def test_output_size():
    Y = rowifilt(barbara_t, (-1,1), (1,-1))
    assert Y.shape[1:] == bshape_double


def test_non_orthogonal_input():
    Y = rowifilt(barbara_t, (1,1), (1,1))
    assert Y.shape[1:] == bshape_double


def test_output_size_non_mult_4():
    Y = rowifilt(barbara_t, (-1,0,0,1), (1,0,0,-1))
    assert Y.shape[1:] == bshape_double


def test_non_orthogonal_input_non_mult_4():
    Y = rowifilt(barbara_t, (1,0,0,1), (1,0,0,1))
    assert Y.shape[1:] == bshape_double


def test_equal_small_in():
    ha = qshift('qshift_b')[0]
    hb = qshift('qshift_b')[1]
    im = barbara[:,0:4,0:4]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = ref_rowifilt(im, ha, hb)
    y_op = rowifilt(im_t, ha, hb)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift1():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    ref = ref_rowifilt(barbara, ha, hb)
    y_op = rowifilt(barbara_t, ha, hb)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift2():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im = barbara[:, :502, :508]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = ref_rowifilt(im, ha, hb)
    y_op = rowifilt(im_t, ha, hb)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

# vim:sw=4:sts=4:et
