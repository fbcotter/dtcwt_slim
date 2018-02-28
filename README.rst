Dual-Tree Complex Wavelet Transform library for Python
======================================================

This library provides support for computing 2D dual-tree complex wavelet
transforms and their inverse using python and tensorflow.

The implementation is designed to be used with batches of multichannel images.
The code can handle 2-D, 3-D or 4-D inputs, but the last 2 dimensions must be
the height and width. In tensorflow terms this equates to using the 'NCHW'
format.

Installation
````````````
The easiest way to install ``dtcwt_slim`` is to clone the repo and pip install
it. Later versions will be released on PyPi but the docs need to updated first::

    $ git clone https://github.com/fbcotter/dtcwt_slim
    $ cd dtcwt_slim
    $ pip install .

    $ python setup.py install

(Although the `develop` command may be more useful if you intend to perform any
significant modification to the library.) A test suite is provided so that you
may verify the code works on your system::

    $ pip install -r tests/requirements.txt
    $ py.test

This will also write test-coverage information to the ``cover/`` directory.

Example Use
```````````
This repo was based off the one here__. There is a tensorflow back-end in there
but conforming to the standards set up in there meant not doing the most
efficient operations in tensorflow (in terms of the shape of inputs). This means
that the interface is almost identical. Below is an example of how to use
dtcwt_slim

.. code python

    import dtcwt_slim
    xfm = dtcwt_slim.Transform2d(biort='near_sym_b', qshift='qshift_b')
    X = tf.placeholder(tf.float32, [None, 3, 512, 512])
    Yl, Yh = xfm.forward(X, nlevels=4) 
    X_hat = xfm.inverse(Yl, Yh)

The key differences between `dtcwt_slim` and `dtcwt` are:

- `dtcwt_slim` doesn't use pyramids. Calling the forward method returns Yl and
  Yh
- Yh and Yscales are returned as lists not tuples, making it easier to
  manipulate individual scales without worrying about immutability
- inverse no longer has the gain_mask input. Applying gains can still be done
  before calling Transform2d.inverse
- `dtcwt_slim` can naturally handle batches of multi-channel signals. E.g.
  inputs of data type 'hw', 'nhw/chw' or 'nchw' are permitted. 
- The highpass outputs have the 6 orientations as the third last dimension now,
  rather than the last. Before an output would be of shape [H, W, 6] but now
  they will be returned as [6, H, W]. If there are multiple channels, then the
  orientations dimension still comes in the third last position - i.e. the
  output will be [C, 6, H, W].
- `dtcwt_slim` can return real highpass outputs instead of complex ones. This
  saves overhead on having to constantly go back to real operations to do things
  that aren't supported in the complex domain (like convolutions). To return
  real highpasses, pass the `complex=False` parameter to the class initializer::

      import dtcwt_slim
      xfm = dtcwt_slim.Transform2d(complex=False)

  The highpass outputs then are a list of length 2.

__ https://github.com/rjw57/dtcwt

Provenance
``````````

Based on the Dual-Tree Complex Wavelet Transform Pack for MATLAB by Nick
Kingsbury, Cambridge University. The original README can be found in
ORIGINAL_README.txt.  This file outlines the conditions of use of the original
MATLAB toolbox.

.. vim:sw=4:sts=4:et
