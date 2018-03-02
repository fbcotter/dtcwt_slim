import pytest

import numpy as np
from dtcwt.numpy import Transform2d as Transform2d_np
from dtcwt_slim.tf import Transform2d
import tests.datasets as datasets

import tensorflow as tf
import py3nvml
PRECISION_DECIMAL = 3


def setup():
    global barbara, barbara_t
    global config, reshape_hp
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

    def new_axes(x):
        if len(x.shape) == 3:
            return [2, 0, 1]
        elif len(x.shape) == 4:
            return [0, 3, 1, 2]
        elif len(x.shape) == 5:
            return [0, 1, 4, 2, 3]
        else:
            return x.shape
    reshape_hp = lambda x: np.transpose(x, new_axes(x))


def test_barbara_loaded():
    assert barbara.shape == (3, 512, 512)
    assert barbara.min() >= 0
    assert barbara.max() <= 1
    assert barbara.dtype == np.float32
    assert barbara_t.get_shape() == (1, 3, 512, 512)


def test_simple():
    xfm = Transform2d()
    Yl, Yh = xfm.forward(barbara_t[0,0])
    assert len(Yl.get_shape()) == 2
    Yl, Yh = xfm.forward(barbara_t[0])
    assert len(Yl.get_shape()) == 3
    Yl, Yh = xfm.forward(barbara_t)
    assert len(Yl.get_shape()) == 4


def test_specific_wavelet():
    xfm = Transform2d(biort='antonini', qshift='qshift_06')
    Yl, Yh = xfm.forward(barbara_t[0,0])
    assert len(Yl.get_shape()) == 2
    Yl, Yh = xfm.forward(barbara_t[0])
    assert len(Yl.get_shape()) == 3
    Yl, Yh = xfm.forward(barbara_t)
    assert len(Yl.get_shape()) == 4


def test_simple_w_scale():
    xfm = Transform2d()
    Yl, Yh, Yscale = xfm.forward(barbara_t, include_scale=True)
    assert len(Yscale) > 0
    for x in Yscale:
        assert x is not None


def test_odd_rows():
    xfm = Transform2d()
    Yl, Yh = xfm.forward(barbara_t[:,:,:509,:], nlevels=4)


def test_odd_cols():
    xfm = Transform2d()
    Yl, Yh = xfm.forward(barbara_t[:,:,:,:509], nlevels=4)


def test_odd_cols_w_scale():
    xfm = Transform2d()
    Yl, Yh, Yscale = xfm.forward(barbara_t[:,:,:,:509], nlevels=4,
                                 include_scale=True)


def test_odd_rows_and_cols():
    xfm = Transform2d()
    Yl, Yh = xfm.forward(barbara_t[:,:,:509,:509], nlevels=4)


def test_rot_symm_modified():
    # This test only checks there is no error running these functions,
    # not that they work
    xfm = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
    Yl, Yh, Yscale = xfm.forward(barbara_t[:,:,:509,:509], nlevels=4,
                                 include_scale=True)


def test_0_levels():
    xfm = Transform2d()
    Yl, Yh = xfm.forward(barbara_t, nlevels=0)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(Yl)[0]
    np.testing.assert_array_almost_equal(out, barbara, PRECISION_DECIMAL)
    assert len(Yh) == 0


def test_0_levels_w_scale():
    xfm = Transform2d()
    Yl, Yh, Yscale = xfm.forward(barbara_t, nlevels=0, include_scale=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(Yl)[0]
    np.testing.assert_array_almost_equal(out, barbara, PRECISION_DECIMAL)
    assert len(Yh) == 0
    assert len(Yscale) == 0


@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_numpy_in(complex):
    X = 100*np.random.randn(100,100)
    xfm = Transform2d(complex=complex)
    Yl, Yh = xfm.forward(X)
    f1 = Transform2d_np()
    p1 = f1.forward(X)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        Yl = sess.run(Yl)
        if complex:
            Yh = sess.run(Yh)
        else:
            Yh = sess.run([s.complex for s in Yh])

    np.testing.assert_array_almost_equal(
        Yl, p1.lowpass, decimal=PRECISION_DECIMAL)
    for x,y in zip(Yh, p1.highpasses):
        np.testing.assert_array_almost_equal(
            x,reshape_hp(y),decimal=PRECISION_DECIMAL)

    X = np.random.randn(100,100)
    Yl, Yh, Yscale = xfm.forward(X, include_scale=True)
    p1 = f1.forward(X, include_scale=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        Yl = sess.run(Yl)
        if complex:
            Yh = sess.run(Yh)
        else:
            Yh = sess.run([s.complex for s in Yh])
        Yscale = sess.run(Yscale)

    np.testing.assert_array_almost_equal(
        Yl, p1.lowpass, decimal=PRECISION_DECIMAL)
    for x,y in zip(Yh, p1.highpasses):
        np.testing.assert_array_almost_equal(
            x,reshape_hp(y),decimal=PRECISION_DECIMAL)
    for x,y in zip(Yscale, p1.scales):
        np.testing.assert_array_almost_equal(
            x,y,decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_numpy_in_batch(complex):
    X = np.random.randn(5,100,100)

    xfm = Transform2d(complex=complex)
    Yl, Yh, Yscale = xfm.forward(X, include_scale=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        Yl = sess.run(Yl)
        if complex:
            Yh = sess.run(Yh)
        else:
            Yh = sess.run([s.complex for s in Yh])
        Yscale = sess.run(Yscale)

    f1 = Transform2d_np()
    for i in range(5):
        p1 = f1.forward(X[i], include_scale=True)
        np.testing.assert_array_almost_equal(
            Yl[i], p1.lowpass, decimal=PRECISION_DECIMAL)
        for x,y in zip(Yh, p1.highpasses):
            np.testing.assert_array_almost_equal(
                x[i], reshape_hp(y), decimal=PRECISION_DECIMAL)
        for x,y in zip(Yscale, p1.scales):
            np.testing.assert_array_almost_equal(
                x[i], y, decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_numpy_batch_ch(complex):
    X = np.random.randn(5,4,100,100)

    xfm = Transform2d(complex=complex)
    Yl, Yh, Yscale = xfm.forward(X, include_scale=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        Yl = sess.run(Yl)
        if complex:
            Yh = sess.run(Yh)
        else:
            Yh = sess.run([s.complex for s in Yh])
        Yscale = sess.run(Yscale)

    f1 = Transform2d_np()
    for i in range(5):
        for j in range(4):
            p1 = f1.forward(X[i,j], include_scale=True)
            np.testing.assert_array_almost_equal(
                Yl[i,j], p1.lowpass, decimal=PRECISION_DECIMAL)
            for x,y in zip(Yh, p1.highpasses):
                np.testing.assert_array_almost_equal(
                    x[i,j], reshape_hp(y), decimal=PRECISION_DECIMAL)
            for x,y in zip(Yscale, p1.scales):
                np.testing.assert_array_almost_equal(
                    x[i,j], y, decimal=PRECISION_DECIMAL)


# Test end to end with numpy inputs
@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_2d_input(complex):
    X = np.random.randn(100,100)
    xfm = Transform2d(complex=complex)
    Yl, Yh = xfm.forward(X)
    x = xfm.inverse(Yl, Yh)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        x_out = sess.run(x)
    np.testing.assert_array_almost_equal(X,x_out,decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_3d_input(complex):
    X = np.random.randn(5,100,100)
    xfm = Transform2d(complex=complex)
    Yl, Yh = xfm.forward(X)
    x = xfm.inverse(Yl, Yh)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        x_out = sess.run(x)
    np.testing.assert_array_almost_equal(X,x_out,decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_4d_input(complex):
    X = np.random.randn(5,4,100,100)
    xfm = Transform2d(complex=complex)
    Yl, Yh = xfm.forward(X)
    x = xfm.inverse(Yl, Yh)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        x_out = sess.run(x)
    np.testing.assert_array_almost_equal(X,x_out,decimal=PRECISION_DECIMAL)


# Test end to end with tf inputs
@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_2d_input_ph(complex):
    X = tf.placeholder(tf.float32, [100,100])
    xfm = Transform2d(complex=complex)
    Yl, Yh = xfm.forward(X)
    x = xfm.inverse(Yl, Yh)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        in_ = np.random.randn(100,100)
        x_out = sess.run(x, {X:in_})
    np.testing.assert_array_almost_equal(in_,x_out,decimal=PRECISION_DECIMAL)


# Test end to end with tf inputs
@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_3d_input_ph(complex):
    X = tf.placeholder(tf.float32, [5, 100,100])
    xfm = Transform2d(complex=complex)
    Yl, Yh = xfm.forward(X)
    x = xfm.inverse(Yl, Yh)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        in_ = np.random.randn(5, 100,100)
        x_out = sess.run(x, {X:in_})
    np.testing.assert_array_almost_equal(in_,x_out,decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("complex", [
    (False), (True)
])
def test_4d_input_ph(complex):
    X = tf.placeholder(tf.float32, [None, 5, 100,100])
    xfm = Transform2d(complex=complex)
    Yl, Yh = xfm.forward(X)
    x = xfm.inverse(Yl, Yh)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        in_ = np.random.randn(4,5, 100,100)
        x_out = sess.run(x, {X:in_})
    np.testing.assert_array_almost_equal(in_,x_out,decimal=PRECISION_DECIMAL)


@pytest.mark.parametrize("complex,test_input,biort,qshift", [
    (True, datasets.mandrill(),'antonini','qshift_a'),
    (False, datasets.mandrill()[100:400,40:460],'legall','qshift_a'),
    (True, datasets.mandrill(),'near_sym_a','qshift_c'),
    (False, datasets.mandrill()[100:375,30:322],'near_sym_b','qshift_d'),
    (True, datasets.mandrill(),'near_sym_b_bp', 'qshift_b_bp')
])
def test_results_match_2d(complex, test_input, biort, qshift):
    """
    Compare forward transform with numpy forward transform for barbara image
    """
    im = test_input
    f_np = Transform2d_np(biort=biort,qshift=qshift)
    p_np = f_np.forward(im, include_scale=True)

    f_tf = Transform2d(biort=biort,qshift=qshift,complex=complex)
    Yl, Yh, Yscale = f_tf.forward(im, include_scale=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        Yl = sess.run(Yl)
        if complex:
            Yh = sess.run(Yh)
        else:
            Yh = sess.run([s.complex for s in Yh])
        Yscale = sess.run(Yscale)

    np.testing.assert_array_almost_equal(
        p_np.lowpass, Yl, decimal=3)
    [np.testing.assert_array_almost_equal(
        reshape_hp(h_np), h_tf, decimal=3) for h_np, h_tf in
        zip(p_np.highpasses, Yh)]
    [np.testing.assert_array_almost_equal(
        s_np, s_tf, decimal=3) for s_np, s_tf in
        zip(p_np.scales, Yscale)]


@pytest.mark.parametrize("test_input,biort,qshift", [
    (datasets.barbara(),'antonini','qshift_a'),
    (datasets.barbara()[100:400,40:460,:],'legall','qshift_a'),
    (datasets.barbara(),'near_sym_a','qshift_c'),
    (datasets.barbara()[100:375,30:322,:],'near_sym_b','qshift_d'),
    (datasets.barbara(),'near_sym_b_bp', 'qshift_b_bp')
])
def test_results_match_3d(test_input, biort, qshift):
    """
    Compare forward transform with numpy forward transform for barbara image
    """
    im = test_input.transpose([2,0,1])
    f_tf = Transform2d(biort=biort,qshift=qshift)
    Yl, Yh, Yscale = f_tf.forward(im, include_scale=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if complex:
            Yl, Yh, Yscale = sess.run([Yl, Yh, Yscale])
        else:
            Yl, Yh, Yscale = sess.run([Yl, [s.complex for s in Yh], Yscale])

    f_np = Transform2d_np(biort=biort,qshift=qshift)
    for i, ch in enumerate(im):
        p_np = f_np.forward(ch, include_scale=True)

        np.testing.assert_array_almost_equal(
            p_np.lowpass, Yl[i], decimal=3)
        [np.testing.assert_array_almost_equal(
            reshape_hp(h_np), h_tf[i], decimal=3) for h_np, h_tf in
            zip(p_np.highpasses, Yh)]
        [np.testing.assert_array_almost_equal(
            s_np, s_tf[i], decimal=3) for s_np, s_tf in
            zip(p_np.scales, Yscale)]


@pytest.mark.parametrize("biort,qshift", [
    ('antonini','qshift_a'),
    ('legall','qshift_a'),
    ('near_sym_a','qshift_c'),
    ('near_sym_b','qshift_d'),
    ('near_sym_b_bp', 'qshift_b_bp')
])
def test_results_match_4d(biort, qshift):
    """
    Compare forward transform with numpy forward transform for barbara image
    """
    im = np.random.randn(5,6,128,128)
    f_tf = Transform2d(biort=biort,qshift=qshift)
    Yl, Yh, Yscale = f_tf.forward(im, include_scale=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if complex:
            Yl, Yh, Yscale = sess.run([Yl, Yh, Yscale])
        else:
            Yl, Yh, Yscale = sess.run([Yl, [s.complex for s in Yh], Yscale])

    f_np = Transform2d_np(biort=biort,qshift=qshift)
    for i, n in enumerate(im):
        for j, ch in enumerate(n):
            p_np = f_np.forward(ch, include_scale=True)

            np.testing.assert_array_almost_equal(
                p_np.lowpass, Yl[i,j], decimal=3)
            [np.testing.assert_array_almost_equal(
                reshape_hp(h_np), h_tf[i,j], decimal=3) for h_np, h_tf in
                zip(p_np.highpasses, Yh)]
            [np.testing.assert_array_almost_equal(
                s_np, s_tf[i,j], decimal=3) for s_np, s_tf in
                zip(p_np.scales, Yscale)]


@pytest.mark.parametrize("test_input,biort,qshift", [
    (datasets.mandrill(),'antonini','qshift_c'),
    (datasets.mandrill()[100:412,40:460],'near_sym_a','qshift_a'),
    (datasets.mandrill(),'legall','qshift_c'),
    (datasets.mandrill()[100:378,20:322],'near_sym_b','qshift_06'),
    (datasets.mandrill(),'near_sym_b_bp', 'qshift_b_bp')
])
def test_results_match_inverse(test_input,biort,qshift):
    im = test_input
    f_np = Transform2d_np(biort=biort, qshift=qshift)
    p_np = f_np.forward(im, nlevels=4, include_scale=True)
    X_np = f_np.inverse(p_np)

    # Use a zero input and the fwd transform to get the shape of
    # the pyramid easily
    f_tf = Transform2d(biort=biort, qshift=qshift)
    Yl, Yh = f_tf.forward(im, nlevels=4)

    # Create ops for the inverse transform
    X_tf = f_tf.inverse(Yl, Yh)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        X_tf = sess.run(X_tf)

    np.testing.assert_array_almost_equal(
        X_np, X_tf, decimal=PRECISION_DECIMAL)


def test_forward_channels():
    batch = 5
    c = 3
    nlevels = 3
    sess = tf.Session()

    ims = np.random.randn(batch, c, 100, 100)
    in_p = tf.placeholder(tf.float32, [None, c, 100, 100])

    # Transform a set of images with forward_channels
    f_tf = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
    Yl, Yh, Yscale = f_tf.forward(
        in_p, nlevels=nlevels, include_scale=True)

    Yl, Yh, Yscale = sess.run([Yl, Yh, Yscale], {in_p: ims})

    # Now do it channel by channel
    in_p2 = tf.placeholder(tf.float32, [batch, 100, 100])
    Yl1, Yh1, Yscale1 = f_tf.forward(in_p2, nlevels=nlevels,
                                     include_scale=True)
    for i in range(c):
        Yl2, Yh2, Yscale2 = sess.run([Yl1, Yh1, Yscale1],
                                     {in_p2: ims[:,i]})
        np.testing.assert_array_almost_equal(
            Yl[:,i], Yl2, decimal=PRECISION_DECIMAL)
        for j in range(nlevels):
            np.testing.assert_array_almost_equal(
                Yh[j][:,i], Yh2[j], decimal=PRECISION_DECIMAL)
            np.testing.assert_array_almost_equal(
                Yscale[j][:,i], Yscale2[j], decimal=PRECISION_DECIMAL)
    sess.close()


#  def test_inverse_channels():
    #  batch = 5
    #  c = 3
    #  nlevels = 3
    #  sess = tf.Session()

    #  # Create the tensors of the right shape by calling the forward function
    #  if data_format == "nhwc":
        #  ims = np.random.randn(batch, 100, 100, c)
        #  in_p = tf.placeholder(tf.float32, [None, 100, 100, c])
        #  f_tf = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
        #  Yl, Yh = unpack(
            #  f_tf.forward_channels(in_p, nlevels=nlevels,
                                  #  data_format=data_format), 'tf')
    #  else:
        #  ims = np.random.randn(batch, c, 100, 100)
        #  in_p = tf.placeholder(tf.float32, [None, c, 100, 100])
        #  f_tf = Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
        #  Yl, Yh = unpack(f_tf.forward_channels(
            #  in_p, nlevels=nlevels, data_format=data_format), 'tf')

    #  # Call the inverse_channels function
    #  start = time.time()
    #  X = f_tf.inverse_channels(Pyramid(Yl, Yh), data_format=data_format)
    #  X, Yl, Yh = sess.run([X, Yl, Yh], {in_p: ims})
    #  print("That took {:.2f}s".format(time.time() - start))

    #  # Now do it channel by channel
    #  in_p2 = tf.zeros((batch, 100, 100), tf.float32)
    #  p_tf = f_tf.forward_channels(in_p2, nlevels=nlevels, data_format="nhw",
                                 #  include_scale=False)
    #  X_t = f_tf.inverse_channels(p_tf,data_format="nhw")
    #  for i in range(c):
        #  Yh1 = []
        #  if data_format == "nhwc":
            #  Yl1 = Yl[:,:,:,i]
            #  for j in range(nlevels):
                #  Yh1.append(Yh[j][:,:,:,i])
        #  else:
            #  Yl1 = Yl[:,i]
            #  for j in range(nlevels):
                #  Yh1.append(Yh[j][:,i])

        #  # Use the eval_inv function to feed the data into the right variables
        #  sess.run(tf.global_variables_initializer())
        #  X1 = sess.run(X_t, {p_tf.lowpass_op: Yl1, p_tf.highpasses_ops: Yh1})

        #  if data_format == "nhwc":
            #  np.testing.assert_array_almost_equal(
                #  X[:,:,:,i], X1, decimal=PRECISION_DECIMAL)
        #  else:
            #  np.testing.assert_array_almost_equal(
                #  X[:,i], X1, decimal=PRECISION_DECIMAL)

    #  sess.close()

#  # vim:sw=4:sts=4:et
