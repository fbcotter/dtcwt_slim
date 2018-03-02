from __future__ import absolute_import

import tensorflow as tf


class ComplexTensor(object):
    """ A wrapper to handle complex tensor as either a complex
    """
    def __init__(self, val):
        # Work out what the form of val is. Is it a pair of real and imaginary?
        # Or a single complex/real number
        if isinstance(val, tuple) or isinstance(val, list):
            assert len(val) == 2
            assert val[0].get_shape().as_list() == val[1].get_shape().as_list()
            self._real = tf.identity(val[0])
            self._imag = tf.identity(val[1])
            self._complex = None
        else:
            self._complex = tf.cast(tf.identity(val), dtype=tf.complex64)
            self._real = tf.real(self._complex)
            self._imag = tf.imag(self._complex)

        self.shape = tuple(self._real.get_shape().as_list())

    @property
    def complex(self):
        if self._complex is None:
            self._complex = tf.complex(self._real, self._imag)
        return self._complex

    @complex.setter
    def complex(self, value):
        self._complex = tf.cast(tf.identity(value), dtype=tf.complex64)
        self._real = tf.real(self._complex)
        self._imag = tf.imag(self._complex)

    @property
    def real(self):
        return self._real

    @real.setter
    def real(self, value):
        self._real = value

    @property
    def imag(self):
        return self._imag

    @imag.setter
    def imag(self, value):
        self._imag = value

    def apply_func(self, f):
        """ Applies the functions independently on real and imaginary components
        """
        return ComplexTensor([f(self.real), f(self.imag)])

    def __add__(self, other):
        if not hasattr(other, 'real'):
            other = ComplexTensor(other)
        return ComplexTensor((self.real+other.real, self.imag+other.imag))

    def __radd__(self, other):
        if not hasattr(other, 'real'):
            other = ComplexTensor(other)
        return ComplexTensor((self.real+other.real, self.imag+other.imag))

    def __sub__(self, other):
        if not hasattr(other, 'real'):
            other = ComplexTensor(other)
        return ComplexTensor((self.real-other.real, self.imag-other.imag))

    def __rsub__(self, other):
        if not hasattr(other, 'real'):
            other = ComplexTensor(other)
        return ComplexTensor((other.real-self.real, other.imag-self.imag))

    def __mul__(self, other):
        if not hasattr(other, 'real'):
            other = ComplexTensor(other)
        return ComplexTensor([self.real*other.real - self.imag*other.imag,
                              self.real*other.imag + self.imag*other.real])

    def __rmul__(self, other):
        # Complex multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ComplexTensor):
            raise NotImplementedError
        else:
            return ComplexTensor([self.real/other, self.imag/other])

    def get_shape(self):
        return self.real.get_shape()

    def __repr__(self):
        return "<ComplexTensor shape={}>".format(self.shape)
