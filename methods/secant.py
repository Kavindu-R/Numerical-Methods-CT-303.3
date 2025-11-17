"""
Secant Method Implementation
============================
File: methods/secant.py

The secant method is similar to Newton-Raphson but doesn't require the derivative.
Instead, it approximates the derivative using a secant line through two points.

Theoretical Background:
- Approximates derivative: f'(x) ≈ [f(x_n) - f(x_{n-1})] / (x_n - x_{n-1})
- Iterative formula: x_{n+1} = x_n - f(x_n) × (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
- Super-linear convergence: order φ ≈ 1.618 (golden ratio!)
- Requires two initial guesses instead of one
- No derivative calculation needed
- Faster than bisection, slightly slower than Newton-Raphson
"""

import numpy as np
from typing import Callable, Tuple, List, Dict


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict:
    """
    Find root of equation f(x) = 0 using secant method.
    
    Parameters:
    -----------
    f : Callable[[float], float]
        The function for which we want to find the root
    x0 : float
        First initial guess
    x1 : float
        Second initial guess
    tol : float, optional
        Tolerance for convergence (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'root': Estimated root value
        - 'iterations': Number of iterations performed
        - 'error': Final error estimate |x_n - x_{n-1}|
        - 'function_value': f(root)
        - 'converged': Boolean indicating if method converged
        - 'history': List of x values at each iteration
        - 'error_history': List of errors at each iteration
        - 'function_evaluations': Total number of function evaluations
    
    Raises:
    -------
    ValueError:
        If initial guesses are identical
        If f(x0) = f(x1) (would cause division by zero)
    
    Example:
    --------
    >>> def f(x): return x**2 - 4
    >>> result = secant(f, 1.0, 3.0)
    >>> print(f"Root: {result['root']:.6f}")
    Root: 2.000000
    """
    
    # Input validation
    if abs(x1 - x0) < 1e-12:
        raise ValueError(
            f"Initial guesses are too close: x0={x0}, x1={x1}. "
            "Secant method requires two distinct initial guesses."
        )
    
    # Initialize tracking variables
    history = [x0, x1]
    error_history = []
    iteration = 0
    function_evaluations = 0
    
    # Evaluate function at initial guesses
    fx0 = f(x0)
    fx1 = f(x1)
    function_evaluations += 2
    
    # Check if initial guesses are roots
    if abs(fx0) < tol:
        return {
            'root': x0,
            'iterations': 0,
            'error': 0.0,
            'function_value': fx0,
            'converged': True,
            'history': [x0],
            'error_history': [0.0],
            'function_evaluations': 1
        }
    
    if abs(fx1) < tol:
        return {
            'root': x1,
            'iterations': 0,
            'error': 0.0,
            'function_value': fx1,
            'converged': True,
            'history': [x1],
            'error_history': [0.0],
            'function_evaluations': 2
        }
    
    # Check if f(x0) = f(x1)
    if abs(fx1 - fx0) < 1e-12:
        raise ValueError(
            f"Function values are identical: f(x0)={fx0:.6e}, f(x1)={fx1:.6e}. "
            "Cannot compute secant line. Try different initial guesses."
        )
    
    # Main secant loop
    while iteration < max_iter:
        # Calculate next approximation using secant formula
        try:
            denominator = fx1 - fx0
            if abs(denominator) < 1e-12:
                return {
                    'root': x1,
                    'iterations': iteration,
                    'error': abs(x1 - x0),
                    'function_value': fx1,
                    'converged': False,
                    'history': history,
                    'error_history': error_history,
                    'function_evaluations': function_evaluations,
                    'failure_reason': 'Denominator too close to zero'
                }
            
            x2 = x1 - fx1 * (x1 - x0) / denominator
            
        except (ZeroDivisionError, OverflowError) as e:
            return {
                'root': x1,
                'iterations': iteration,
                'error': abs(x1 - x0),
                'function_value': fx1,
                'converged': False,
                'history': history,
                'error_history': error_history,
                'function_evaluations': function_evaluations,
                'failure_reason': f'Numerical error: {str(e)}'
            }
        
        # Evaluate function at new point
        fx2 = f(x2)
        function_evaluations += 1
        
        # Calculate error
        error = abs(x2 - x1)
        error_history.append(error)
        history.append(x2)
        
        # Check convergence
        if abs(fx2) < tol or error < tol:
            return {
                'root': x2,
                'iterations': iteration + 1,
                'error': error,
                'function_value': fx2,
                'converged': True,
                'history': history,
                'error_history': error_history,
                'function_evaluations': function_evaluations
            }
        
        # Update for next iteration
        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2
        
        iteration += 1
    
    # Maximum iterations reached
    return {
        'root': x2,
        'iterations': max_iter,
        'error': error_history[-1] if error_history else float('inf'),
        'function_value': fx2,
        'converged': False,
        'history': history,
        'error_history': error_history,
        'function_evaluations': function_evaluations,
        'failure_reason': 'Maximum iterations reached'
    }


def secant_verbose(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict:
    """
    Secant method with detailed iteration output (for debugging/learning).
    
    Same parameters and returns as secant(), but prints iteration details.
    """
    
    print("\n" + "="*85)
    print("SECANT METHOD - DETAILED OUTPUT")
    print("="*85)
    print(f"Initial guesses: x₀ = {x0}, x₁ = {x1}")
    print(f"Tolerance: {tol}")
    print(f"Maximum iterations: {max_iter}")
    print(f"f(x₀) = {f(x0):.6f}")
    print(f"f(x₁) = {f(x1):.6f}")
    print("="*85)
    print(f"{'Iter':<6} {'x_{n-1}':<15} {'x_n':<15} {'x_{n+1}':<15} {'f(x_{n+1})':<15} {'Error':<15}")
    print("-"*85)
    
    # Input validation
    if abs(x1 - x0) < 1e-12:
        raise ValueError("Initial guesses are too close")
    
    history = [x0, x1]
    error_history = []
    iteration = 0
    function_evaluations = 0
    
    fx0 = f(x0)
    fx1 = f(x1)
    function_evaluations += 2
    
    if abs(fx1 - fx0) < 1e-12:
        raise ValueError("Function values are identical at initial guesses")
    
    print(f"{0:<6} {x0:<15.10f} {x1:<15.10f} {'---':<15} {fx1:<15.6e} {'---':<15}")
    
    while iteration < max_iter:
        denominator = fx1 - fx0
        if abs(denominator) < 1e-12:
            print("\n⚠ Warning: Denominator too close to zero!")
            break
        
        x2 = x1 - fx1 * (x1 - x0) / denominator
        fx2 = f(x2)
        function_evaluations += 1
        
        error = abs(x2 - x1)
        error_history.append(error)
        history.append(x2)
        
        print(f"{iteration+1:<6} {x0:<15.10f} {x1:<15.10f} {x2:<15.10f} {fx2:<15.6e} {error:<15.6e}")
        
        if abs(fx2) < tol or error < tol:
            print("-"*85)
            print(f" CONVERGED in {iteration + 1} iterations")
            print(f" Root found: x = {x2:.10f}")
            print(f" f(root) = {fx2:.2e}")
            print(f" Final error: {error:.2e}")
            print(f" Function evaluations: {function_evaluations}")
            print(f" Convergence rate: Super-linear (order φ ≈ 1.618)")
            print("="*85 + "\n")
            
            return {
                'root': x2,
                'iterations': iteration + 1,
                'error': error,
                'function_value': fx2,
                'converged': True,
                'history': history,
                'error_history': error_history,
                'function_evaluations': function_evaluations
            }
        
        x0, x1 = x1, x2
        fx0, fx1 = fx1, fx2
        
        iteration += 1
    
    print("-"*85)
    print(f"✗ Maximum iterations ({max_iter}) reached")
    print(f"✗ Best estimate: x = {x2:.10f}")
    print(f"✗ f(x) = {fx2:.2e}")
    print(f"✗ Final error: {error:.2e}")
    print(f"✗ Function evaluations: {function_evaluations}")
    print("="*85 + "\n")
    
    return {
        'root': x2,
        'iterations': max_iter,
        'error': error,
        'function_value': fx2,
        'converged': False,
        'history': history,
        'error_history': error_history,
        'function_evaluations': function_evaluations
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Secant Method Implementation")
    print("=" * 85)
    
    # Test Case 1: Simple quadratic
    print("\nTest 1: f(x) = x² - 4, Root: x = 2")
    def f1(x):
        return x**2 - 4
    
    result1 = secant_verbose(f1, 1.0, 3.0, tol=1e-6)
    
    # Test Case 2: Cubic equation
    print("\nTest 2: f(x) = x³ - x - 2, Root: x ≈ 1.52138")
    def f2(x):
        return x**3 - x - 2
    
    result2 = secant_verbose(f2, 1.0, 2.0, tol=1e-6)
    
    # Test Case 3: Transcendental equation
    print("\nTest 3: f(x) = cos(x) - x, Root: x ≈ 0.73909")
    def f3(x):
        return np.cos(x) - x
    
    result3 = secant_verbose(f3, 0.0, 1.0, tol=1e-6)
    
    # Test Case 4: Exponential equation
    print("\nTest 4: f(x) = e^x - 3x, Root: x ≈ 1.512")
    def f4(x):
        return np.exp(x) - 3*x
    
    result4 = secant_verbose(f4, 1.0, 2.0, tol=1e-6)
    
    # Test Case 5: Polynomial
    print("\nTest 5: f(x) = x³ - 2x - 5, Root: x ≈ 2.09455")
    def f5(x):
        return x**3 - 2*x - 5
    
    result5 = secant_verbose(f5, 2.0, 3.0, tol=1e-6)
    
    print("\n" + "="*85)
    print("SUMMARY OF ALL TESTS")
    print("="*85)
    print(f"{'Test':<8} {'Root':<18} {'Iterations':<12} {'Function Evals':<15}")
    print("-"*85)
    print(f"{'Test 1':<8} {result1['root']:<18.10f} {result1['iterations']:<12} {result1['function_evaluations']:<15}")
    print(f"{'Test 2':<8} {result2['root']:<18.10f} {result2['iterations']:<12} {result2['function_evaluations']:<15}")
    print(f"{'Test 3':<8} {result3['root']:<18.10f} {result3['iterations']:<12} {result3['function_evaluations']:<15}")
    print(f"{'Test 4':<8} {result4['root']:<18.10f} {result4['iterations']:<12} {result4['function_evaluations']:<15}")
    print(f"{'Test 5':<8} {result5['root']:<18.10f} {result5['iterations']:<12} {result5['function_evaluations']:<15}")
    print("="*85)
    
    print("\n KEY INSIGHTS:")
    print(" Secant method converges faster than Bisection")
    print(" No derivative needed (unlike Newton-Raphson)")
    print(" Super-linear convergence (order φ ≈ 1.618 - Golden Ratio!)")
    print(" Uses fewer function evaluations than finite difference Newton-Raphson")
    print(" Great balance between speed and simplicity!")