import pytest

import numpy as np
from dtcwt_slim.numpy import Transform2d as Transform2d_np
from dtcwt_slim.torch.transform2d import DTCWTForward, DTCWTInverse
import tests.datasets as datasets

import torch
import py3nvml
PRECISION_DECIMAL = 3


def setup():
    global barbara, barbara_t
    global bshape, bshape_half
    global ref_coldfilt, ch
    py3nvml.grab_gpus(1, gpu_fraction=0.5)
    barbara = datasets.barbara()
    barbara = (barbara/barbara.max()).astype('float32')
    barbara = barbara.transpose([2, 0, 1])
    bshape = list(barbara.shape)
    bshape_half = bshape[:]
    bshape_half[1] //= 2
    barbara_t = torch.unsqueeze(torch.tensor(barbara, dtype=torch.float32),
                                dim=0)
    ch = barbara_t.shape[1]

    # Some useful functions
    ref_coldfilt = lambda x, ha, hb: np.stack(
        [np_coldfilt(s, ha, hb) for s in x], axis=0)


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert list(barbara_t.shape) == [1, 3, 512, 512]


def test_simple():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yhr, Yhi = xfm(barbara_t)
    assert len(Yl.shape) == 4


def test_specific_wavelet():
    xfm = DTCWTForward(C=3, J=3, biort='antonini', qshift='qshift_06')
    Yl, Yhr, Yhi = xfm(barbara_t)
    assert len(Yl.shape) == 4


def test_odd_rows():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yhr, Yhi = xfm(barbara_t[:,:,:509])


def test_odd_cols():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yhr, Yhi = xfm(barbara_t[:,:,:,:509])


def test_odd_rows_and_cols():
    xfm = DTCWTForward(C=3, J=3)
    Yl, Yhr, Yhi = xfm(barbara_t[:,:,:509,:509])


#  def test_rot_symm_modified():
    #  # This test only checks there is no error running these functions,
    #  # not that they work
    #  xfm = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
    #  Yl, Yh, Yscale = xfm.forward(barbara_t[:,:,:509,:509], nlevels=4,
                                 #  include_scale=True)


#  def test_0_levels():
    #  xfm = Transform2d()
    #  Yl, Yh = xfm.forward(barbara_t, nlevels=0)
    #  with tf.Session(config=config) as sess:
        #  sess.run(tf.global_variables_initializer())
        #  out = sess.run(Yl)[0]
    #  np.testing.assert_array_almost_equal(out, barbara, PRECISION_DECIMAL)
    #  assert len(Yh) == 0


#  def test_0_levels_w_scale():
    #  xfm = Transform2d()
    #  Yl, Yh, Yscale = xfm.forward(barbara_t, nlevels=0, include_scale=True)
    #  with tf.Session(config=config) as sess:
        #  sess.run(tf.global_variables_initializer())
        #  out = sess.run(Yl)[0]
    #  np.testing.assert_array_almost_equal(out, barbara, PRECISION_DECIMAL)
    #  assert len(Yh) == 0
    #  assert len(Yscale) == 0


def test_numpy_in():
    X = 100*np.random.randn(3, 5, 100, 100)
    xfm = DTCWTForward(C=5, J=3)
    Yl, Yhr, Yhi = xfm(torch.tensor(X, dtype=torch.float32))
    f1 = Transform2d_np()
    yl, yh = f1.forward(X)

    np.testing.assert_array_almost_equal(
        Yl, yl, decimal=PRECISION_DECIMAL)
    for i in range(len(yh)):
        np.testing.assert_array_almost_equal(
            Yhr[i], yh[i].real, decimal=PRECISION_DECIMAL)
        np.testing.assert_array_almost_equal(
            Yhi[i], yh[i].imag, decimal=PRECISION_DECIMAL)


# Test end to end with numpy inputs
def test_end2end():
    X = 100*np.random.randn(3, 5, 100, 100)
    xfm = DTCWTForward(C=5, J=3)
    Yl, Yhr, Yhi = xfm(torch.tensor(X, dtype=torch.float32))
    ifm = DTCWTInverse(C=5, J=3)
    y = ifm(Yl, Yhr, Yhi)
    np.testing.assert_array_almost_equal(X,y,decimal=PRECISION_DECIMAL)
    X = 100*np.random.randn(3, 5, 200, 200)
    xfm = DTCWTForward(C=5, J=6)
    Yl, Yhr, Yhi = xfm(torch.tensor(X, dtype=torch.float32))
    ifm = DTCWTInverse(C=5, J=6)
    y = ifm(Yl, Yhr, Yhi)
    np.testing.assert_array_almost_equal(X,y,decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("biort,qshift,size", [
    ('antonini','qshift_a', (128,128)),
    ('legall','qshift_a', (99,100)),
    ('near_sym_a','qshift_c', (104, 101)),
    ('near_sym_b','qshift_d', (126, 126)),
])
def test_results_match_4d(biort, qshift, size):
    im = np.random.randn(5,6,*size)
    xfm = DTCWTForward(C=6, J=4, biort=biort, qshift=qshift)
    Yl, Yhr, Yhi = xfm(torch.tensor(im, dtype=torch.float32))
    ifm = DTCWTInverse(C=6, J=4, biort=biort, qshift=qshift)
    y = ifm(Yl, Yhr, Yhi)
    np.testing.assert_array_almost_equal(im,y,decimal=PRECISION_DECIMAL)

    f_np = Transform2d_np(biort=biort,qshift=qshift)
    yl, yh = f_np.forward(im, nlevels=4)
    np.testing.assert_array_almost_equal(yl, Yl)
    for i in range(len(yh)):
        np.testing.assert_array_almost_equal(Yhr[i], yh[i].real,
                                             decimal=PRECISION_DECIMAL)
        np.testing.assert_array_almost_equal(Yhi[i], yh[i].imag,
                                             decimal=PRECISION_DECIMAL)
