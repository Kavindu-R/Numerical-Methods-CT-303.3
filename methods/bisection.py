import numpy as np
from typing import Callable, Tuple, List, Dict


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict:
    """
    Find root of equation f(x) = 0 using bisection method.
    
    Parameters:
    -----------
    f : Callable[[float], float]
        The function for which we want to find the root
    a : float
        Left endpoint of initial interval
    b : float
        Right endpoint of initial interval
    tol : float, optional
        Tolerance for convergence (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'root': Estimated root value
        - 'iterations': Number of iterations performed
        - 'error': Final error estimate (interval width)
        - 'function_value': f(root)
        - 'converged': Boolean indicating if method converged
        - 'history': List of midpoint values at each iteration
        - 'error_history': List of errors at each iteration
    
    Raises:
    -------
    ValueError:
        If f(a) and f(b) have the same sign (no root in interval)
        If a >= b (invalid interval)
    
    Example:
    --------
    >>> def f(x): return x**2 - 4
    >>> result = bisection(f, 0, 3)
    >>> print(f"Root: {result['root']:.6f}")
    Root: 2.000000
    """
    
    # Input validation
    if a >= b:
        raise ValueError(f"Invalid interval: a={a} must be less than b={b}")
    
    # Evaluate function at endpoints
    fa = f(a)
    fb = f(b)
    
    # Check if root exists in interval
    if fa * fb > 0:
        raise ValueError(
            f"Function has same sign at endpoints: f({a})={fa:.6f}, f({b})={fb:.6f}. "
            "Bisection method requires f(a) and f(b) to have opposite signs."
        )
    
    # Check if endpoints are roots
    if abs(fa) < tol:
        return {
            'root': a,
            'iterations': 0,
            'error': 0.0,
            'function_value': fa,
            'converged': True,
            'history': [a],
            'error_history': [0.0]
        }
    
    if abs(fb) < tol:
        return {
            'root': b,
            'iterations': 0,
            'error': 0.0,
            'function_value': fb,
            'converged': True,
            'history': [b],
            'error_history': [0.0]
        }
    
    # Initialize tracking variables
    iteration = 0
    history = []
    error_history = []
    
    # Main bisection loop
    while iteration < max_iter:
        # Calculate midpoint
        c = (a + b) / 2.0
        fc = f(c)
        
        # Store history
        history.append(c)
        error = (b - a) / 2.0
        error_history.append(error)
        
        # Check convergence
        if abs(fc) < tol or error < tol:
            return {
                'root': c,
                'iterations': iteration + 1,
                'error': error,
                'function_value': fc,
                'converged': True,
                'history': history,
                'error_history': error_history
            }
        
        # Update interval
        if fa * fc < 0:
            # Root is in left half
            b = c
            fb = fc
        else:
            # Root is in right half
            a = c
            fa = fc
        
        iteration += 1
    
    # Maximum iterations reached
    c = (a + b) / 2.0
    fc = f(c)
    error = (b - a) / 2.0
    
    return {
        'root': c,
        'iterations': max_iter,
        'error': error,
        'function_value': fc,
        'converged': False,
        'history': history,
        'error_history': error_history
    }


def bisection_verbose(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict:
    """
    Bisection method with detailed iteration output (for debugging/learning).
    
    Same parameters and returns as bisection(), but prints iteration details.
    """
    
    print("\n" + "="*70)
    print("BISECTION METHOD - DETAILED OUTPUT")
    print("="*70)
    print(f"Initial interval: [{a}, {b}]")
    print(f"Tolerance: {tol}")
    print(f"Maximum iterations: {max_iter}")
    print(f"f(a) = f({a}) = {f(a):.6f}")
    print(f"f(b) = f({b}) = {f(b):.6f}")
    print("="*70)
    print(f"{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'Error':<12}")
    print("-"*70)
    
    # Input validation
    if a >= b:
        raise ValueError(f"Invalid interval: a={a} must be less than b={b}")
    
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        raise ValueError("Function has same sign at endpoints")
    
    iteration = 0
    history = []
    error_history = []
    
    while iteration < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        error = (b - a) / 2.0
        
        history.append(c)
        error_history.append(error)
        
        # Print iteration details
        print(f"{iteration+1:<6} {a:<12.6f} {b:<12.6f} {c:<12.6f} {fc:<12.6f} {error:<12.6e}")
        
        if abs(fc) < tol or error < tol:
            print("-"*70)
            print(f"✓ CONVERGED in {iteration + 1} iterations")
            print(f"✓ Root found: x = {c:.10f}")
            print(f"✓ f(root) = {fc:.2e}")
            print(f"✓ Final error: {error:.2e}")
            print("="*70 + "\n")
            
            return {
                'root': c,
                'iterations': iteration + 1,
                'error': error,
                'function_value': fc,
                'converged': True,
                'history': history,
                'error_history': error_history
            }
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        
        iteration += 1
    
    print("-"*70)
    print(f"✗ Maximum iterations ({max_iter}) reached")
    print(f"✗ Best estimate: x = {c:.10f}")
    print(f"✗ f(x) = {fc:.2e}")
    print(f"✗ Final error: {error:.2e}")
    print("="*70 + "\n")
    
    c = (a + b) / 2.0
    fc = f(c)
    error = (b - a) / 2.0
    
    return {
        'root': c,
        'iterations': max_iter,
        'error': error,
        'function_value': fc,
        'converged': False,
        'history': history,
        'error_history': error_history
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Bisection Method Implementation")
    print("=" * 70)
    
    # Test Case 1: Simple quadratic
    print("\nTest 1: f(x) = x² - 4, Root: x = 2")
    def f1(x):
        return x**2 - 4
    
    result1 = bisection_verbose(f1, 0, 3, tol=1e-6)
    
    # Test Case 2: Cubic equation
    print("\nTest 2: f(x) = x³ - x - 2, Root: x ≈ 1.52138")
    def f2(x):
        return x**3 - x - 2
    
    result2 = bisection_verbose(f2, 1, 2, tol=1e-6)
    
    # Test Case 3: Transcendental equation
    print("\nTest 3: f(x) = cos(x) - x, Root: x ≈ 0.73909")
    def f3(x):
        return np.cos(x) - x
    
    result3 = bisection_verbose(f3, 0, 1, tol=1e-6)
    
    print("\n" + "="*70)
    print("SUMMARY OF ALL TESTS")
    print("="*70)
    print(f"Test 1: Root = {result1['root']:.10f}, Iterations = {result1['iterations']}")
    print(f"Test 2: Root = {result2['root']:.10f}, Iterations = {result2['iterations']}")
    print(f"Test 3: Root = {result3['root']:.10f}, Iterations = {result3['iterations']}")
    print("="*70)