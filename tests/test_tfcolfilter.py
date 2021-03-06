import numpy as np
from dtcwt_slim.coeffs import biort, qshift
from dtcwt.numpy.lowlevel import colfilter as np_colfilter

import tests.datasets as datasets
from dtcwt_slim.tf.lowlevel import colfilter
import tensorflow as tf
import py3nvml


def setup():
    global barbara, barbara_t, tf
    global bshape, bshape_extrarow
    global ref_colfilter, t_shape, config
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_extrarow = bshape[:]
    bshape_extrarow[1] += 1
    barbara_t = tf.expand_dims(tf.constant(barbara, dtype=tf.float32),
                               axis=0)

    # Some useful functions
    ref_colfilter = lambda x, h: np.stack(
        [np_colfilter(s, h) for s in x], axis=0)
    t_shape = lambda x: x.get_shape().as_list()


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert barbara_t.get_shape() == (1, 3, 512, 512)


def test_odd_size():
    y_op = colfilter(barbara_t, [-1,2,-1])
    assert t_shape(y_op)[1:] == bshape


def test_even_size():
    y_op = colfilter(barbara_t, [-1,-1])
    assert t_shape(y_op)[1:] == bshape_extrarow


def test_qshift():
    h = qshift('qshift_a')[0]
    y_op = colfilter(barbara_t, h)
    assert t_shape(y_op)[1:] == bshape_extrarow


def test_biort():
    h = biort('antonini')[0]
    y_op = colfilter(barbara_t, h)
    assert t_shape(y_op)[1:] == bshape


def test_even_size_batch():
    zero_t = tf.zeros([1, *barbara.shape], tf.float32)
    y_op = colfilter(zero_t, [-1,1])
    assert t_shape(y_op)[1:] == bshape_extrarow
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    assert not np.any(y[:] != 0.0)


def test_equal_small_in():
    h = qshift('qshift_b')[0]
    im = barbara[:,0:4,0:4]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = ref_colfilter(im, h)
    y_op = colfilter(im_t, h)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_biort1():
    h = biort('near_sym_b')[0]
    ref = ref_colfilter(barbara, h)
    y_op = colfilter(barbara_t, h)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_biort2():
    h = biort('near_sym_b')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = ref_colfilter(im, h)
    y_op = colfilter(im_t, h)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift1():
    h = qshift('qshift_c')[0]
    ref = ref_colfilter(barbara, h)
    y_op = colfilter(barbara_t, h)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift2():
    h = qshift('qshift_c')[0]
    im = barbara[:, 52:407, 30:401]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = ref_colfilter(im, h)
    y_op = colfilter(im_t, h)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

# vim:sw=4:sts=4:et
