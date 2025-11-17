"""
Numerical Methods Package for Root Finding
This package contains implementations of numerical methods for root finding.
"""

from .bisection import bisection, bisection_verbose
from .newton_raphson import (
    newton_raphson, 
    newton_raphson_verbose,
    newton_raphson_auto_derivative
)
from .secant import secant, secant_verbose

__all__ = [
    'bisection',
    'bisection_verbose',
    'newton_raphson',
    'newton_raphson_verbose',
    'newton_raphson_auto_derivative',
    'secant',
    'secant_verbose'
]

__version__ = '1.0.0'
__author__ = 'Group CT-303.3 Numerical Methods'