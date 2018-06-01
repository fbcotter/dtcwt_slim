import pytest

import numpy as np
from dtcwt.numpy import Transform2d as Transform2d_np, Pyramid
from dtcwt_slim.numpy import Transform2d


def test_3d():
    xfm_np = Transform2d_np()
    xfm = Transform2d()

    a = np.random.randn(5,100,100)
    Yl, Yh = xfm.forward(a)
    for i in range(a.shape[0]):
        p = xfm_np.forward(a[i])
        np.testing.assert_array_equal(p.lowpass, Yl[i])
        for j in range(len(Yh)):
            np.testing.assert_array_equal(
                np.transpose(p.highpasses[j], (2, 0, 1)), Yh[j][i])

    X = xfm.inverse(Yl, Yh)
    for i in range(a.shape[0]):
        p = Pyramid(Yl[i], [np.transpose(Yh[j][i], (1,2,0))
                            for j in range(len(Yh))])
        X_np = xfm_np.inverse(p)
        np.testing.assert_array_equal(X_np, X[i])


def test_2d():
    xfm_np = Transform2d_np()
    xfm = Transform2d()

    a = np.random.randn(100,100)
    Yl, Yh = xfm.forward(a)
    p = xfm_np.forward(a)
    np.testing.assert_array_equal(p.lowpass, Yl)
    for j in range(len(Yh)):
        np.testing.assert_array_equal(
            np.transpose(p.highpasses[j], (2, 0, 1)), Yh[j])

    X = xfm.inverse(Yl, Yh)
    p = Pyramid(Yl, [np.transpose(Yh[j], (1,2,0)) for j in range(len(Yh))])
    X_np = xfm_np.inverse(p)
    np.testing.assert_array_equal(X_np, X)


def test_4d():
    xfm_np = Transform2d_np()
    xfm = Transform2d()

    a = np.random.randn(10, 5, 100,100)
    Yl, Yh = xfm.forward(a)
    for n in range(a.shape[0]):
        for i in range(a.shape[1]):
            p = xfm_np.forward(a[n,i])
            np.testing.assert_array_equal(p.lowpass, Yl[n,i])
            for j in range(len(Yh)):
                np.testing.assert_array_equal(
                    np.transpose(p.highpasses[j], (2, 0, 1)), Yh[j][n,i])

    X = xfm.inverse(Yl, Yh)
    for n in range(a.shape[0]):
        for i in range(a.shape[1]):
            p = Pyramid(Yl[n,i], [np.transpose(Yh[j][n,i], (1,2,0))
                                  for j in range(len(Yh))])
            X_np = xfm_np.inverse(p)
            np.testing.assert_array_equal(X_np, X[n,i])
