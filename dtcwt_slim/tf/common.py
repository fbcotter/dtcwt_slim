from __future__ import absolute_import

import tensorflow as tf


class ComplexTensor(tf.Tensor):
    """ A wrapper to handle complex tensor as a pair of real and imaginary
    numbers.
    """
    def __init__(self, val):
        # Work out what the form of val is. Is it a pair of real and imaginary?
        # Or a single complex/real number
        if isinstance(val, ComplexTensor):
            return val
        elif isinstance(val, tuple) or isinstance(val, list):
            assert len(val) == 2
            try:
                assert (val[0].get_shape().as_list() ==
                        val[1].get_shape().as_list())
            # Will get an attribute error if they are numpy arrays not tf ones
            except AttributeError:
                assert val[0].shape == val[1].shape
            self.realimag = True
            self._real = tf.identity(val[0])
            self._imag = tf.identity(val[1])
            self._complex = tf.complex(self._real, self._imag)
        else:
            self.realimag = False
            self._complex = tf.cast(tf.identity(val), dtype=tf.complex64)
            self._real = None
            self._imag = None

        super().__init__(op=self.complex.op, value_index=0, dtype=tf.complex64)

        self._norm = None
        self._norm2 = None
        self._phase = None

    @property
    def complex(self):
        if self._complex is None:
            self._complex = tf.complex(self._real, self._imag)
        return self._complex

    #  @complex.setter
    #  def complex(self, value):
        #  self._complex = tf.cast(tf.identity(value), dtype=tf.complex64)
        #  self._op = self.complex.op
        #  self.realimag = False
        #  self._real = None
        #  self._imag = None

    @property
    def real(self):
        if self._real is None:
            self._real = tf.real(self._complex)
        return self._real

    #  @real.setter
    #  def real(self, value):
        #  self._real = value
        #  self._complex = tf.complex(self._real, self.imag)
        #  self._op = self.complex.op

    @property
    def imag(self):
        if self._imag is None:
            self._imag = tf.imag(self._complex)
        return self._imag

    #  @imag.setter
    #  def imag(self, value):
        #  self._imag = value
        #  self._complex = None
        #  self._op = self.complex.op

    @property
    def norm(self):
        if self._norm is None:
            self._norm = tf.sqrt(self.norm2)
        return self._norm

    @property
    def norm2(self):
        """ Returns the norm squared
        """
        if self._norm2 is None:
            self._norm2 = self.real**2 + self.imag**2
        return self._norm2

    @property
    def phase(self):
        if self._phase is None:
            self._phase = tf.angle(self.complex)
        return self._phase

    def apply_func(self, f):
        """ Applies the functions independently on real and imaginary components
        then returns a complex tensor instance.
        """
        return ComplexTensor([f(self.real), f(self.imag)])

    def __add__(self, other):
        if self.realimag:
            if not hasattr(other, 'real'):
                other = ComplexTensor(other)
            return ComplexTensor((self.real+other.real, self.imag+other.imag))
        else:
            return ComplexTensor(self.complex+other)

    def __radd__(self, other):
        if self.realimag:
            if not hasattr(other, 'real'):
                other = ComplexTensor(other)
            return ComplexTensor((self.real+other.real, self.imag+other.imag))
        else:
            return ComplexTensor(self.complex+other)

    def __sub__(self, other):
        if self.realimag:
            if not hasattr(other, 'real'):
                other = ComplexTensor(other)
            return ComplexTensor((self.real-other.real, self.imag-other.imag))
        else:
            return ComplexTensor(self.complex-other)

    def __rsub__(self, other):
        if self.realimag:
            if not hasattr(other, 'real'):
                other = ComplexTensor(other)
            return ComplexTensor((other.real-self.real, other.imag-self.imag))
        else:
            return ComplexTensor(other-self.complex)

    def __mul__(self, other):
        if tf.identity(other).dtype == tf.float32:
            return ComplexTensor([self.real*other, self.imag*other])
        else:
            if self.realimag:
                if not hasattr(other, 'real'):
                    other = ComplexTensor(other)
                return ComplexTensor([
                    self.real*other.real - self.imag*other.imag,
                    self.real*other.imag + self.imag*other.real])
            else:
                return ComplexTensor(self.complex * other)

    def __rmul__(self, other):
        # Complex multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ComplexTensor):
            raise NotImplementedError
        else:
            if self.realimag:
                return ComplexTensor([self.real/other, self.imag/other])
            else:
                return ComplexTensor(self.complex/other)

    def get_shape(self):
        return self.real.get_shape()

    def __repr__(self):
        return "<ComplexTensor shape={}>".format(self.shape)
