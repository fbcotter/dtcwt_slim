import os
from pytest import raises

import numpy as np
from dtcwt.compat import dtwavexfm2, dtwaveifm2
from dtcwt.coeffs import biort, qshift
import tests.datasets as datasets

TOLERANCE = 1e-12

def setup():
    global mandrill
    mandrill = datasets.mandrill()

def test_mandrill_loaded():
    assert mandrill.shape == (512, 512)
    assert mandrill.min() >= 0
    assert mandrill.max() <= 1
    assert mandrill.dtype == np.float32

def test_simple():
    Yl, Yh = dtwavexfm2(mandrill)

def test_specific_wavelet():
    Yl, Yh = dtwavexfm2(mandrill, biort=biort('antonini'), qshift=qshift('qshift_06'))

def test_1d():
    Yl, Yh = dtwavexfm2(mandrill[0,:])

def test_3d():
    with raises(ValueError):
        Yl, Yh = dtwavexfm2(np.dstack((mandrill, mandrill)))

def test_simple_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill, include_scale=True)

    assert len(Yscale) > 0
    for x in Yscale:
        assert x is not None

def test_odd_rows():
    Yl, Yh = dtwavexfm2(mandrill[:509,:])

def test_odd_rows_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:], include_scale=True)

def test_odd_cols():
    Yl, Yh = dtwavexfm2(mandrill[:,:509])

def test_odd_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:509], include_scale=True)

def test_odd_rows_and_cols():
    Yl, Yh = dtwavexfm2(mandrill[:,:509])

def test_odd_rows_and_cols_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill[:509,:509], include_scale=True)

def test_rot_symm_modified():
    # This test only checks there is no error running these functions, not that they work
    Yl, Yh, Yscale = dtwavexfm2(mandrill, biort='near_sym_b_bp', qshift='qshift_b_bp', include_scale=True)
    Z = dtwaveifm2(Yl, Yh, biort='near_sym_b_bp', qshift='qshift_b_bp')

def test_0_levels():
    Yl, Yh = dtwavexfm2(mandrill, nlevels=0)
    assert np.all(np.abs(Yl - mandrill) < TOLERANCE)
    assert len(Yh) == 0

def test_0_levels_w_scale():
    Yl, Yh, Yscale = dtwavexfm2(mandrill, nlevels=0, include_scale=True)
    assert np.all(np.abs(Yl - mandrill) < TOLERANCE)
    assert len(Yh) == 0
    assert len(Yscale) == 0

def test_integer_input():
    # Check that an integer input is correctly coerced into a floating point
    # array
    Yl, Yh = dtwavexfm2([[1,2,3,4], [1,2,3,4]])
    assert np.any(Yl != 0)

def test_integer_perfect_recon():
    # Check that an integer input is correctly coerced into a floating point
    # array and reconstructed
    A = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int32)
    Yl, Yh = dtwavexfm2(A)
    B = dtwaveifm2(Yl, Yh)
    assert np.max(np.abs(A-B)) < 1e-5

def test_float32_input():
    # Check that an float32 input is correctly output as float32
    Yl, Yh = dtwavexfm2(mandrill.astype(np.float32))
    assert np.issubsctype(Yl.dtype, np.float32)
    assert np.all(list(np.issubsctype(x.dtype, np.complex64) for x in Yh))

# vim:sw=4:sts=4:et
