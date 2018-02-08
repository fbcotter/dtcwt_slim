from pytest import raises

import numpy as np
from dtcwt_slim.coeffs import qshift
from dtcwt_slim.numpy.lowlevel import coldfilt as np_coldfilt

import tests.datasets as datasets
from dtcwt_slim.tf.lowlevel import coldfilt
import tensorflow as tf
import py3nvml


def setup():
    global barbara, barbara_t, tf
    global bshape, bshape_half
    global ref_coldfilt, t_shape, config
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[1] //= 2
    barbara_t = tf.expand_dims(tf.constant(barbara, dtype=tf.float32),
                               axis=0)

    # Some useful functions
    ref_coldfilt = lambda x, ha, hb: np.stack(
        [np_coldfilt(s, ha, hb) for s in x], axis=0)
    t_shape = lambda x: x.get_shape().as_list()


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert barbara_t.get_shape() == (1, 3, 512, 512)


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
    assert t_shape(y_op)[1:] == bshape_half


def test_equal_small_in():
    ha = qshift('qshift_b')[0]
    hb = qshift('qshift_b')[1]
    im = barbara[:,0:4,0:4]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = ref_coldfilt(im, ha, hb)
    y_op = coldfilt(im_t, ha, hb)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift1():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    ref = ref_coldfilt(barbara, ha, hb)
    y_op = coldfilt(barbara_t, ha, hb)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)


def test_equal_numpy_qshift2():
    ha = qshift('qshift_c')[0]
    hb = qshift('qshift_c')[1]
    im = barbara[:, :508, :502]
    im_t = tf.expand_dims(tf.constant(im, tf.float32), axis=0)
    ref = ref_coldfilt(im, ha, hb)
    y_op = coldfilt(im_t, ha, hb)
    with tf.Session(config=config) as sess:
        y = sess.run(y_op)
    np.testing.assert_array_almost_equal(y[0], ref, decimal=4)

# vim:sw=4:sts=4:et
