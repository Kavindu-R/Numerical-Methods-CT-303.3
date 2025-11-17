"""
Test Functions and Equations Module.
This module contains all test equations used for comparing numerical methods.
Each equation includes the function, its derivative, known roots, and metadata.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple


class TestEquation:
    """
    Class to represent a test equation with all necessary information.
    """
    
    def __init__(
        self,
        name: str,
        f: Callable[[float], float],
        df: Callable[[float], float],
        known_roots: List[float],
        bisection_interval: Tuple[float, float],
        newton_guess: float,
        secant_guesses: Tuple[float, float],
        description: str = ""
    ):
        """
        Initialize a test equation.
        
        Parameters:
        -----------
        name : str
            Name/label for the equation
        f : Callable
            The function f(x)
        df : Callable
            The derivative f'(x)
        known_roots : List[float]
            List of known exact or approximate roots
        bisection_interval : Tuple[float, float]
            Initial interval [a, b] for bisection method
        newton_guess : float
            Initial guess for Newton-Raphson method
        secant_guesses : Tuple[float, float]
            Two initial guesses for secant method
        description : str
            Description of the equation
        """
        self.name = name
        self.f = f
        self.df = df
        self.known_roots = known_roots
        self.bisection_interval = bisection_interval
        self.newton_guess = newton_guess
        self.secant_guesses = secant_guesses
        self.description = description
    
    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return f"TestEquation(name='{self.name}', roots={self.known_roots})"



# TEST EQUATION DEFINITIONS


def get_test_equations() -> Dict[str, TestEquation]:
    """
    Get dictionary of all test equations.
    
    Returns:
    --------
    Dict[str, TestEquation]
        Dictionary mapping equation IDs to TestEquation objects
    """
    
    equations = {}
    
    # Equation 1
    # f(x) = x³ - 6x² + 11x - 6
    equations['eq1'] = TestEquation(
        name="Equation 1 - f(x) = x³ - 6x² + 11x - 6",
        f=lambda x: x**3 - 6*x**2 + 11*x - 6,
        df=lambda x: 3*x**2 - 12*x + 11,
        known_roots=[1.0, 2.0, 3.0],
        bisection_interval=(0.5, 3.5),
        newton_guess=2.5,
        secant_guesses=(1.0, 3.0),
        description="f(x) = x³ - 6x² + 11x - 6"
    )
    
    # Equation 2
    # f(x) = cos(x) - x
    equations['eq2'] = TestEquation(
        name="Equation 2 - f(x) = cos(x) - x",
        f=lambda x: np.cos(x) - x,
        df=lambda x: -np.sin(x) - 1,
        known_roots=[0.739],
        bisection_interval=(0, 1),
        newton_guess=0.5,
        secant_guesses=(0.0, 1.0),
        description="f(x) = cos(x) - x"
    )
    
   # Equation 3
   # f(x) = e^x - 3x²
    equations['eq3'] = TestEquation(
        name="Equation 3 - f(x) = e^x - 3x²",
        f=lambda x: np.exp(x) - 3*(x**2),
        df=lambda x: np.exp(x) - 6*x,
        known_roots=[0.620],
        bisection_interval=(0, 1),
        newton_guess=0.6,
        secant_guesses=(0.4, 0.8),
        description="f(x) = e^x - 3x²"
    )
    
    
    return equations


def get_equation_by_id(eq_id: str) -> TestEquation:
    """
    Get a specific test equation by its ID.
    
    Parameters:
    -----------
    eq_id : str
        Equation identifier (e.g., 'eq1', 'eq2', ...)
    
    Returns:
    --------
    TestEquation
        The requested test equation
    """
    equations = get_test_equations()
    if eq_id not in equations:
        raise ValueError(f"Unknown equation ID: {eq_id}. Available: {list(equations.keys())}")
    return equations[eq_id]


def list_all_equations():
    """
    Print a list of all available test equations.
    """
    equations = get_test_equations()
    
    print("\n" + "="*80)
    print("AVAILABLE TEST EQUATIONS")
    print("="*80)
    print(f"{'ID':<8} {'Name':<25} {'Description':<45}")
    print("-"*80)
    
    for eq_id, eq in equations.items():
        print(f"{eq_id:<8} {eq.name:<25} {eq.description:<45}")
    
    print("="*80)
    print(f"Total equations: {len(equations)}\n")


def validate_equation(eq: TestEquation, tolerance: float = 1e-6):
    """
    Validate that known roots are actually roots of the equation.
    
    Parameters:
    -----------
    eq : TestEquation
        Equation to validate
    tolerance : float
        Tolerance for root validation
    """
    print(f"\nValidating equation: {eq.name}")
    print(f"Description: {eq.description}")
    print("-" * 60)
    
    all_valid = True
    for i, root in enumerate(eq.known_roots, 1):
        f_value = eq.f(root)
        is_valid = abs(f_value) < tolerance
        status = "✓" if is_valid else "✗"
        
        print(f"Root {i}: x = {root:.10f}")
        print(f"  f(x) = {f_value:.2e} {status}")
        
        if not is_valid:
            all_valid = False
    
    if all_valid:
        print("\n✓ All roots validated successfully!")
    else:
        print("\n✗ Some roots failed validation!")
    
    return all_valid


def validate_all_equations():
    """
    Validate all test equations.
    """
    equations = get_test_equations()
    
    print("\n" + "="*80)
    print("VALIDATING ALL TEST EQUATIONS")
    print("="*80)
    
    results = {}
    for eq_id, eq in equations.items():
        results[eq_id] = validate_equation(eq)
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    for eq_id, valid in results.items():
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"{eq_id}: {status}")
    
    print("="*80 + "\n")
    
    return all(results.values())



# EXAMPLE USAGE

if __name__ == "__main__":
    print("Test Functions Module")
    print("=" * 80)
    
    # List all equations
    list_all_equations()
    
    # Validate all equations
    all_valid = validate_all_equations()
    
    # Example: Get and test a specific equation
    print("\n" + "="*80)
    print("EXAMPLE: Working with a specific equation")
    print("="*80)
    
    eq = get_equation_by_id('eq3')
    print(f"\nEquation: {eq.name}")
    print(f"Description: {eq.description}")
    print(f"Known root: {eq.known_roots[0]}")
    print(f"Bisection interval: {eq.bisection_interval}")
    print(f"Newton initial guess: {eq.newton_guess}")
    print(f"Secant initial guesses: {eq.secant_guesses}")
    
    # Test the function
    x_test = eq.known_roots[0]
    print(f"\nTesting at known root x = {x_test}:")
    print(f"f({x_test}) = {eq.f(x_test):.2e}")
    print(f"f'({x_test}) = {eq.df(x_test):.6f}")
    
    # Plot range example
    print("\n" + "="*80)
    print("Suggested plotting ranges for visualization:")
    print("="*80)
    
    for eq_id, eq in get_test_equations().items():
        a, b = eq.bisection_interval
        margin = (b - a) * 0.2
        print(f"{eq_id} ({eq.name}): [{a-margin:.2f}, {b+margin:.2f}]")
    
    print("="*80)