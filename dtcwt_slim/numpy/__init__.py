"""
A backend which uses NumPy to perform the filtering. This backend should always
be available.

"""

from .transform2d import Transform2d

__all__ = [
    'Transform2d',
]
