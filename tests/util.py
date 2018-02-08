import functools
import numpy as np
import pytest

from dtcwt.opencl.lowlevel import _HAVE_CL as HAVE_CL
#  from dtcwt.utils import _HAVE_TF as HAVE_TF
from dtcwt.tf.lowlevel import _HAVE_TF as HAVE_TF

from six.moves import xrange

TOLERANCE = 1e-6

def assert_almost_equal(a, b, tolerance=TOLERANCE):
    md = np.abs(a-b).max()
    if md <= tolerance:
        return

    raise AssertionError(
            'Arrays differ by a maximum of {0} which is greater than the tolerance of {1}'.
            format(md, tolerance))

def assert_percentile_almost_equal(a, b, percentile=90, tolerance=TOLERANCE):
    md = np.percentile(np.abs(a-b), percentile)
    if md <= tolerance:
        return

    raise AssertionError(
            'Arrays differ by a maximum of {0} in the {2}th percentile which is greater than the tolerance of {1}'.
            format(md, tolerance, percentile))

def _mean(a, axis=None, *args, **kwargs):
    """Equivalent to numpy.mean except that the axis along which the mean is taken is not removed."""

    rv = np.mean(a, axis=axis, *args, **kwargs)

    if axis is not None:
        rv = np.expand_dims(rv, axis)

    return rv

def centre_indices(ndim=2,apron=8):
    """Returns the centre indices for the correct number of dimension
    """
    return tuple([slice(apron,-apron) for i in xrange(ndim)])

def summarise_mat(M, apron=8):
    """HACK to provide a 'summary' matrix consisting of the corners of the
    matrix and summed versions of the sub matrices.

    N.B. Keep this in sync with matlab/verif_m_to_npz.py.

    """
    centre = M[apron:-apron,apron:-apron,...]
    centre_sum = _mean(_mean(centre, axis=0), axis=1)

    return np.vstack((
        np.hstack((M[:apron,:apron,...], _mean(M[:apron,apron:-apron,...], axis=1), M[:apron,-apron:,...])),
        np.hstack((_mean(M[apron:-apron,:apron,...], axis=0), centre_sum, _mean(M[apron:-apron,-apron:,...], axis=0))),
        np.hstack((M[-apron:,:apron,...], _mean(M[-apron:,apron:-apron,...], axis=1), M[-apron:,-apron:,...])),
    ))

def summarise_cube(M, apron=4):
    """Provide a summary cube, extending  summarise_mat to 3D
    """
    return np.dstack(
        [summarise_mat(M[:,:,i,...], apron) for i in xrange(M.shape[-2])]
    )

skip_if_no_cl = pytest.mark.skipif(not HAVE_CL, reason="OpenCL not present")
skip_if_no_tf = pytest.mark.skipif(not HAVE_TF, reason="Tensorflow not present")

