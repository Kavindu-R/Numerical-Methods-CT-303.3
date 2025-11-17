"""
Main Execution Script.
This script runs all numerical methods on all test equations
and generates comparison results.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List
from tabulate import tabulate

# Import methods
from methods import bisection, newton_raphson, secant

# Import test functions
from test_functions import get_test_equations, list_all_equations

# Import configuration
import config


def run_single_test(method_name: str, method_func, eq_id: str, eq) -> Dict:
    """
    Run a single numerical method on a single equation.
    
    Parameters:
    -----------
    method_name : str
        Name of the method
    method_func : callable
        Method function to call
    eq_id : str
        Equation identifier
    eq : TestEquation
        Test equation object
    
    Returns:
    --------
    dict : Results dictionary
    """
    
    try:
        # Start timing
        start_time = time.time()
        
        # Run appropriate method
        if method_name == 'bisection':
            a, b = eq.bisection_interval
            result = method_func(
                eq.f, a, b,
                tol=config.DEFAULT_TOLERANCE,
                max_iter=config.MAX_ITERATIONS
            )
        
        elif method_name == 'newton_raphson':
            result = method_func(
                eq.f, eq.df, eq.newton_guess,
                tol=config.DEFAULT_TOLERANCE,
                max_iter=config.MAX_ITERATIONS
            )
        
        elif method_name == 'secant':
            x0, x1 = eq.secant_guesses
            result = method_func(
                eq.f, x0, x1,
                tol=config.DEFAULT_TOLERANCE,
                max_iter=config.MAX_ITERATIONS
            )
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate error from known root
        if eq.known_roots:
            known_root = eq.known_roots[0]  # Use first known root
            actual_error = abs(result['root'] - known_root)
        else:
            actual_error = None
        
        # Prepare output
        output = {
            'equation_id': eq_id,
            'equation_name': eq.name,
            'method': method_name,
            'root': result['root'],
            'iterations': result['iterations'],
            'final_error': result['error'],
            'function_value': result['function_value'],
            'converged': result['converged'],
            'execution_time': execution_time,
            'actual_error': actual_error,
            'known_root': eq.known_roots[0] if eq.known_roots else None
        }
        
        # Add function evaluations if available (secant method)
        if 'function_evaluations' in result:
            output['function_evaluations'] = result['function_evaluations']
        else:
            # Estimate for other methods
            if method_name == 'bisection':
                output['function_evaluations'] = result['iterations'] + 2
            elif method_name == 'newton_raphson':
                output['function_evaluations'] = result['iterations'] * 2  # f and f'
        
        return output
    
    except Exception as e:
        print(f"  ✗ Error in {method_name} for {eq_id}: {str(e)}")
        return {
            'equation_id': eq_id,
            'equation_name': eq.name,
            'method': method_name,
            'root': None,
            'iterations': None,
            'final_error': None,
            'function_value': None,
            'converged': False,
            'execution_time': None,
            'actual_error': None,
            'known_root': eq.known_roots[0] if eq.known_roots else None,
            'error_message': str(e)
        }


def run_all_tests() -> pd.DataFrame:
    """
    Run all methods on all test equations.
    
    Returns:
    --------
    pd.DataFrame : Results dataframe
    """
    
    print("\n" + "="*80)
    print("RUNNING ALL NUMERICAL METHODS ON ALL TEST EQUATIONS")
    print("="*80)
    
    equations = get_test_equations()
    methods = {
        'bisection': bisection,
        'newton_raphson': newton_raphson,
        'secant': secant
    }
    
    results = []
    total_tests = len(equations) * len(methods)
    test_count = 0
    
    for eq_id, eq in equations.items():
        print(f"\n Testing Equation: {eq.name} ({eq.description})")
        print(f"   Known root: {eq.known_roots[0] if eq.known_roots else 'Unknown'}")
        print("-" * 80)
        
        for method_name, method_func in methods.items():
            test_count += 1
            print(f"  [{test_count}/{total_tests}] Running {method_name}...", end=" ")
            
            result = run_single_test(method_name, method_func, eq_id, eq)
            results.append(result)
            
            if result['converged']:
                print(f"✓ Converged in {result['iterations']} iterations")
            else:
                print(f"✗ Failed to converge")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETED")
    print("="*80)
    
    return df


def display_results_summary(df: pd.DataFrame):
    """
    Display a summary of results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    """
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Summary by method
    print("\n Summary by Method:")
    print("-" * 80)
    
    summary_data = []
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        
        summary_data.append({
            'Method': method.replace('_', ' ').title(),
            'Success Rate': f"{method_df['converged'].sum()}/{len(method_df)}",
            'Avg Iterations': f"{method_df['iterations'].mean():.2f}",
            'Avg Time (ms)': f"{method_df['execution_time'].mean() * 1000:.4f}",
            'Avg Error': f"{method_df['final_error'].mean():.2e}"
        })
    
    print(tabulate(summary_data, headers='keys', tablefmt='grid'))
    
    # Detailed results table
    print("\n Detailed Results:")
    print("-" * 80)
    
    display_df = df[[
        'equation_name', 'method', 'root', 'iterations', 
        'final_error', 'converged', 'execution_time'
    ]].copy()
    
    display_df['method'] = display_df['method'].str.replace('_', ' ').str.title()
    display_df['root'] = display_df['root'].apply(lambda x: f"{x:.10f}" if x is not None else "N/A")
    display_df['final_error'] = display_df['final_error'].apply(
        lambda x: f"{x:.2e}" if x is not None else "N/A"
    )
    display_df['execution_time'] = display_df['execution_time'].apply(
        lambda x: f"{x*1000:.4f} ms" if x is not None else "N/A"
    )
    
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))


def save_results(df: pd.DataFrame):
    """
    Save results to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    """
    
    output_file = f"{config.TABLES_DIR}/{config.RESULTS_CSV_NAME}"
    df.to_csv(output_file, index=False)
    print(f"\n Results saved to: {output_file}")


def compare_methods(df: pd.DataFrame):
    """
    Compare methods across different metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    """
    
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)
    
    # Comparison by equation
    equations = df['equation_name'].unique()
    
    for eq_name in equations:
        eq_df = df[df['equation_name'] == eq_name]
        
        print(f"\n🔍 {eq_name}:")
        print("-" * 80)
        
        comparison_data = []
        for _, row in eq_df.iterrows():
            comparison_data.append({
                'Method': row['method'].replace('_', ' ').title(),
                'Root': f"{row['root']:.10f}" if row['root'] is not None else "N/A",
                'Iterations': row['iterations'] if row['iterations'] is not None else "N/A",
                'Error': f"{row['final_error']:.2e}" if row['final_error'] is not None else "N/A",
                'Time (ms)': f"{row['execution_time']*1000:.4f}" if row['execution_time'] is not None else "N/A",
                'Status': "✓" if row['converged'] else "✗"
            })
        
        print(tabulate(comparison_data, headers='keys', tablefmt='grid'))


def main():
    """
    Main execution function.
    """
    
    print("\n" + "="*80)
    print(f"  {config.PROJECT_NAME}")
    print(f"  {config.COURSE_CODE}")
    print("="*80)
    
    # Print configuration
    if config.VERBOSE:
        config.print_config()
    
    # List available equations
    list_all_equations()
    
    # Run all tests
    results_df = run_all_tests()
    
    # Display results
    display_results_summary(results_df)
    
    # Compare methods
    compare_methods(results_df)
    
    # Save results
    save_results(results_df)
    
    print("\n" + "="*80)
    print(" EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\n Next steps:")
    print("  1. Check the results CSV in results/tables/")
    print("  2. Run analysis scripts to generate plots")
    print("  3. Review convergence patterns")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()