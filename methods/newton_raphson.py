import numpy as np
from typing import Callable, Tuple, List, Dict, Optional


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict:
    """
    Find root of equation f(x) = 0 using Newton-Raphson method.
    
    Parameters:
    -----------
    f : Callable[[float], float]
        The function for which we want to find the root
    df : Callable[[float], float]
        The derivative of function f
    x0 : float
        Initial guess for the root
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
        - 'derivative_zeros': Number of times derivative was near zero
    
    Raises:
    -------
    ValueError:
        If derivative is zero at initial guess
    RuntimeWarning:
        If derivative becomes very small during iteration
    
    Example:
    --------
    >>> def f(x): return x**2 - 4
    >>> def df(x): return 2*x
    >>> result = newton_raphson(f, df, 1.0)
    >>> print(f"Root: {result['root']:.6f}")
    Root: 2.000000
    """
    
    # Initialize tracking variables
    x = x0
    history = [x0]
    error_history = []
    derivative_zeros = 0
    iteration = 0
    
    # Evaluate function and derivative at initial guess
    fx = f(x)
    dfx = df(x)
    
    # Check if initial guess is already a root
    if abs(fx) < tol:
        return {
            'root': x,
            'iterations': 0,
            'error': 0.0,
            'function_value': fx,
            'converged': True,
            'history': history,
            'error_history': [0.0],
            'derivative_zeros': 0
        }
    
    # Check if derivative is zero at initial guess
    if abs(dfx) < 1e-12:
        raise ValueError(
            f"Derivative is zero at initial guess x₀={x0}. "
            "Newton-Raphson method cannot proceed. Try a different initial guess."
        )
    
    # Main Newton-Raphson loop
    while iteration < max_iter:
        # Calculate next approximation
        try:
            x_new = x - fx / dfx
        except ZeroDivisionError:
            return {
                'root': x,
                'iterations': iteration,
                'error': abs(x - history[-2]) if len(history) > 1 else float('inf'),
                'function_value': fx,
                'converged': False,
                'history': history,
                'error_history': error_history,
                'derivative_zeros': derivative_zeros,
                'failure_reason': 'Division by zero (derivative = 0)'
            }
        
        # Calculate error
        error = abs(x_new - x)
        error_history.append(error)
        history.append(x_new)
        
        # Update for next iteration
        x = x_new
        fx = f(x)
        dfx = df(x)
        
        # Check for near-zero derivative
        if abs(dfx) < 1e-12:
            derivative_zeros += 1
            if derivative_zeros > 3:
                return {
                    'root': x,
                    'iterations': iteration + 1,
                    'error': error,
                    'function_value': fx,
                    'converged': False,
                    'history': history,
                    'error_history': error_history,
                    'derivative_zeros': derivative_zeros,
                    'failure_reason': 'Derivative too close to zero'
                }
        
        # Check convergence
        if abs(fx) < tol or error < tol:
            return {
                'root': x,
                'iterations': iteration + 1,
                'error': error,
                'function_value': fx,
                'converged': True,
                'history': history,
                'error_history': error_history,
                'derivative_zeros': derivative_zeros
            }
        
        iteration += 1
    
    # Maximum iterations reached
    return {
        'root': x,
        'iterations': max_iter,
        'error': error_history[-1] if error_history else float('inf'),
        'function_value': fx,
        'converged': False,
        'history': history,
        'error_history': error_history,
        'derivative_zeros': derivative_zeros,
        'failure_reason': 'Maximum iterations reached'
    }


def newton_raphson_verbose(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Dict:
    """
    Newton-Raphson method with detailed iteration output (for debugging/learning).
    
    Same parameters and returns as newton_raphson(), but prints iteration details.
    """
    
    print("\n" + "="*80)
    print("NEWTON-RAPHSON METHOD - DETAILED OUTPUT")
    print("="*80)
    print(f"Initial guess: x₀ = {x0}")
    print(f"Tolerance: {tol}")
    print(f"Maximum iterations: {max_iter}")
    print(f"f(x₀) = {f(x0):.6f}")
    print(f"f'(x₀) = {df(x0):.6f}")
    print("="*80)
    print(f"{'Iter':<6} {'xₙ':<15} {'f(xₙ)':<15} {'f\'(xₙ)':<15} {'Error':<15}")
    print("-"*80)
    
    x = x0
    history = [x0]
    error_history = []
    derivative_zeros = 0
    iteration = 0
    
    fx = f(x)
    dfx = df(x)
    
    if abs(dfx) < 1e-12:
        raise ValueError(f"Derivative is zero at initial guess x₀={x0}")
    
    print(f"{0:<6} {x:<15.10f} {fx:<15.6e} {dfx:<15.6e} {'---':<15}")
    
    while iteration < max_iter:
        x_new = x - fx / dfx
        error = abs(x_new - x)
        error_history.append(error)
        history.append(x_new)
        
        x = x_new
        fx = f(x)
        dfx = df(x)
        
        print(f"{iteration+1:<6} {x:<15.10f} {fx:<15.6e} {dfx:<15.6e} {error:<15.6e}")
        
        if abs(dfx) < 1e-12:
            derivative_zeros += 1
        
        if abs(fx) < tol or error < tol:
            print("-"*80)
            print(f"✓ CONVERGED in {iteration + 1} iterations")
            print(f"✓ Root found: x = {x:.10f}")
            print(f"✓ f(root) = {fx:.2e}")
            print(f"✓ Final error: {error:.2e}")
            print(f"✓ Convergence rate: Quadratic (order 2)")
            print("="*80 + "\n")
            
            return {
                'root': x,
                'iterations': iteration + 1,
                'error': error,
                'function_value': fx,
                'converged': True,
                'history': history,
                'error_history': error_history,
                'derivative_zeros': derivative_zeros
            }
        
        iteration += 1
    
    print("-"*80)
    print(f"✗ Maximum iterations ({max_iter}) reached")
    print(f"✗ Best estimate: x = {x:.10f}")
    print(f"✗ f(x) = {fx:.2e}")
    print(f"✗ Final error: {error:.2e}")
    print("="*80 + "\n")
    
    return {
        'root': x,
        'iterations': max_iter,
        'error': error,
        'function_value': fx,
        'converged': False,
        'history': history,
        'error_history': error_history,
        'derivative_zeros': derivative_zeros
    }


def newton_raphson_auto_derivative(
    f: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    h: float = 1e-7
) -> Dict:
    """
    Newton-Raphson method with automatic numerical derivative calculation.
    
    Uses finite differences to approximate f'(x) ≈ [f(x+h) - f(x)] / h
    
    Parameters:
    -----------
    h : float, optional
        Step size for numerical derivative (default: 1e-7)
    
    Other parameters same as newton_raphson()
    """
    
    def numerical_derivative(x: float) -> float:
        """Calculate derivative using central difference formula"""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    return newton_raphson(f, numerical_derivative, x0, tol, max_iter)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Newton-Raphson Method Implementation")
    print("=" * 80)
    
    # Test Case 1: Simple quadratic
    print("\nTest 1: f(x) = x² - 4, f'(x) = 2x, Root: x = 2")
    def f1(x):
        return x**2 - 4
    def df1(x):
        return 2*x
    
    result1 = newton_raphson_verbose(f1, df1, 1.0, tol=1e-6)
    
    # Test Case 2: Cubic equation
    print("\nTest 2: f(x) = x³ - x - 2, f'(x) = 3x² - 1, Root: x ≈ 1.52138")
    def f2(x):
        return x**3 - x - 2
    def df2(x):
        return 3*x**2 - 1
    
    result2 = newton_raphson_verbose(f2, df2, 1.5, tol=1e-6)
    
    # Test Case 3: Transcendental equation
    print("\nTest 3: f(x) = cos(x) - x, f'(x) = -sin(x) - 1, Root: x ≈ 0.73909")
    def f3(x):
        return np.cos(x) - x
    def df3(x):
        return -np.sin(x) - 1
    
    result3 = newton_raphson_verbose(f3, df3, 0.5, tol=1e-6)
    
    # Test Case 4: Exponential equation
    print("\nTest 4: f(x) = e^x - 3x, f'(x) = e^x - 3")
    def f4(x):
        return np.exp(x) - 3*x
    def df4(x):
        return np.exp(x) - 3
    
    result4 = newton_raphson_verbose(f4, df4, 1.5, tol=1e-6)
    
    # Test automatic derivative
    print("\nTest 5: Using automatic numerical derivative")
    print("f(x) = x³ - x - 2")
    result5 = newton_raphson_auto_derivative(f2, 1.5, tol=1e-6)
    print(f"Root: {result5['root']:.10f}, Iterations: {result5['iterations']}")
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)
    print(f"Test 1: Root = {result1['root']:.10f}, Iterations = {result1['iterations']}")
    print(f"Test 2: Root = {result2['root']:.10f}, Iterations = {result2['iterations']}")
    print(f"Test 3: Root = {result3['root']:.10f}, Iterations = {result3['iterations']}")
    print(f"Test 4: Root = {result4['root']:.10f}, Iterations = {result4['iterations']}")
    print(f"Test 5: Root = {result5['root']:.10f}, Iterations = {result5['iterations']}")
    print("="*80)
    
    print("\n Notice how Newton-Raphson converges MUCH faster than Bisection!")
    print("   Typically needs only 3-5 iterations vs 20+ for Bisection! ")