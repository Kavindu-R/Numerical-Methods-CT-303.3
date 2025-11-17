"""
Test Functions Package.
Package containing test equations for numerical methods.
"""

from .equations import (
    TestEquation,
    get_test_equations,
    get_equation_by_id,
    list_all_equations,
    validate_equation,
    validate_all_equations
)

__all__ = [
    'TestEquation',
    'get_test_equations',
    'get_equation_by_id',
    'list_all_equations',
    'validate_equation',
    'validate_all_equations'
]