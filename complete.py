"""
Complete Output Generator
Run this script to generate ALL outputs, plots, tables, and logs!
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import project modules
from methods import bisection, newton_raphson, secant
from test_functions import get_test_equations, validate_all_equations
from analysis.visualizations import (
    plot_convergence_history,
    plot_function_with_roots,
    plot_iterations_comparison,
    plot_execution_time_comparison,
    generate_all_plots
)
import config


def setup_logging():
    """
    Setup logging to both file and console.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOGS_DIR, f"analysis_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def run_method_verbose(method_name, method_func, eq_id, eq):
    """
    Run a method and collect detailed information.
    """
    result = {}
    
    try:
        start_time = time.time()
        
        if method_name == 'bisection':
            a, b = eq.bisection_interval
            output = method_func(eq.f, a, b, tol=config.DEFAULT_TOLERANCE, max_iter=config.MAX_ITERATIONS)
        elif method_name == 'newton_raphson':
            output = method_func(eq.f, eq.df, eq.newton_guess, tol=config.DEFAULT_TOLERANCE, max_iter=config.MAX_ITERATIONS)
        elif method_name == 'secant':
            x0, x1 = eq.secant_guesses
            output = method_func(eq.f, x0, x1, tol=config.DEFAULT_TOLERANCE, max_iter=config.MAX_ITERATIONS)
        
        end_time = time.time()
        
        result = {
            'equation_id': eq_id,
            'equation_name': eq.name,
            'equation_description': eq.description,
            'method': method_name,
            'root_found': output['root'],
            'known_root': eq.known_roots[0] if eq.known_roots else None,
            'iterations': output['iterations'],
            'final_error': output['error'],
            'function_value': output['function_value'],
            'converged': output['converged'],
            'execution_time_ms': (end_time - start_time) * 1000,
            'history': output.get('history', []),
            'error_history': output.get('error_history', [])
        }
        
        # Calculate actual error if known root exists
        if eq.known_roots:
            result['actual_error'] = abs(output['root'] - eq.known_roots[0])
        
        # Add function evaluations
        if 'function_evaluations' in output:
            result['function_evaluations'] = output['function_evaluations']
        else:
            if method_name == 'bisection':
                result['function_evaluations'] = output['iterations'] + 2
            elif method_name == 'newton_raphson':
                result['function_evaluations'] = output['iterations'] * 2
        
        logging.info(f"✓ {method_name.upper()} on {eq.name}: Root={output['root']:.10f}, Iterations={output['iterations']}")
        
    except Exception as e:
        logging.error(f"✗ {method_name.upper()} on {eq.name}: {str(e)}")
        result = {
            'equation_id': eq_id,
            'equation_name': eq.name,
            'equation_description': eq.description,
            'method': method_name,
            'error_message': str(e),
            'converged': False
        }
    
    return result


def generate_all_outputs():
    """
    Generate ALL outputs: tables, plots, and logs!
    """
    
    print("\n" + "="*80)
    print(" COMPLETE ANALYSIS - GENERATING ALL OUTPUTS")
    print("="*80)
    
    # Setup logging
    log_file = setup_logging()
    logging.info("="*80)
    logging.info("Starting Complete Analysis")
    logging.info(f"Log file: {log_file}")
    logging.info("="*80)
    
    # Create all directories
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Step 1: Validate equations
    print("\n STEP 1: Validating Test Equations")
    print("-" * 80)
    logging.info("\n=== VALIDATING TEST EQUATIONS ===")
    validate_all_equations()
    
    # Step 2: Run all methods on all equations
    print("\n STEP 2: Running All Methods on All Equations")
    print("-" * 80)
    logging.info("\n=== RUNNING ALL METHODS ===")
    
    equations = get_test_equations()
    methods = {
        'bisection': bisection,
        'newton_raphson': newton_raphson,
        'secant': secant
    }
    
    all_results = []
    total_tests = len(equations) * len(methods)
    test_count = 0
    
    for eq_id, eq in equations.items():
        print(f"\n🔬 Testing: {eq.name} - {eq.description}")
        logging.info(f"\n--- Equation: {eq.name} ({eq.description}) ---")
        
        for method_name, method_func in methods.items():
            test_count += 1
            print(f"  [{test_count}/{total_tests}] {method_name.ljust(20)}", end=" ... ")
            
            result = run_method_verbose(method_name, method_func, eq_id, eq)
            all_results.append(result)
            
            if result.get('converged', False):
                print(f"✓ {result['iterations']} iterations")
            else:
                print(f"✗ Failed")
    
    # Step 3: Create comprehensive DataFrame
    print("\n STEP 3: Creating Results DataFrame")
    print("-" * 80)
    logging.info("\n=== CREATING RESULTS DATAFRAME ===")
    
    df = pd.DataFrame(all_results)
    
    # Save main results CSV
    main_csv = os.path.join(config.TABLES_DIR, "complete_results.csv")
    df.to_csv(main_csv, index=False)
    print(f"✓ Saved: {main_csv}")
    logging.info(f"Saved main results: {main_csv}")
    
    # Step 4: Generate Summary Tables
    print("\n STEP 4: Generating Summary Tables")
    print("-" * 80)
    logging.info("\n=== GENERATING SUMMARY TABLES ===")
    
    # Summary by method
    summary_by_method = df.groupby('method').agg({
        'iterations': ['mean', 'std', 'min', 'max'],
        'execution_time_ms': ['mean', 'std'],
        'final_error': ['mean', 'min', 'max'],
        'converged': 'sum'
    }).round(6)
    
    summary_method_file = os.path.join(config.TABLES_DIR, "summary_by_method.csv")
    summary_by_method.to_csv(summary_method_file)
    print(f"✓ Saved: {summary_method_file}")
    logging.info(f"Saved method summary: {summary_method_file}")
    
    # Summary by equation
    summary_by_equation = df.groupby('equation_name').agg({
        'iterations': 'mean',
        'execution_time_ms': 'mean',
        'converged': 'sum'
    }).round(6)
    
    summary_eq_file = os.path.join(config.TABLES_DIR, "summary_by_equation.csv")
    summary_by_equation.to_csv(summary_eq_file)
    print(f"✓ Saved: {summary_eq_file}")
    logging.info(f"Saved equation summary: {summary_eq_file}")
    
    # Detailed comparison table
    comparison_df = df.pivot_table(
        values='iterations',
        index='equation_name',
        columns='method',
        aggfunc='first'
    )
    
    comparison_file = os.path.join(config.TABLES_DIR, "iterations_comparison.csv")
    comparison_df.to_csv(comparison_file)
    print(f"✓ Saved: {comparison_file}")
    logging.info(f"Saved comparison table: {comparison_file}")
    
    # Step 5: Generate ALL Plots
    print("\n STEP 5: Generating All Plots")
    print("-" * 80)
    logging.info("\n=== GENERATING ALL PLOTS ===")
    
    # Convergence plots for each equation
    print("\n   Convergence History Plots...")
    for eq_id, eq in equations.items():
        print(f"    • {eq.name}")
        logging.info(f"Generating convergence plot: {eq.name}")
        try:
            plot_convergence_history(eq, save=True)
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception as e:
            logging.error(f"Error plotting {eq.name}: {str(e)}")
    
    # Function plots
    print("\n   Function Plots with Roots...")
    for eq_id, eq in equations.items():
        print(f"    • {eq.name}")
        logging.info(f"Generating function plot: {eq.name}")
        try:
            plot_function_with_roots(eq, save=True)
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception as e:
            logging.error(f"Error plotting function {eq.name}: {str(e)}")
    
    # Comparison plots
    print("\n   Comparison Plots...")
    print("    • Iterations Comparison")
    logging.info("Generating iterations comparison plot")
    try:
        plot_iterations_comparison(df, save=True)
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception as e:
        logging.error(f"Error plotting iterations comparison: {str(e)}")
    
    print("    • Execution Time Comparison")
    logging.info("Generating execution time comparison plot")
    try:
        plot_execution_time_comparison(df, save=True)
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception as e:
        logging.error(f"Error plotting time comparison: {str(e)}")
    
    # Step 6: Generate Text Report
    print("\n STEP 6: Generating Text Report")
    print("-" * 80)
    logging.info("\n=== GENERATING TEXT REPORT ===")
    
    report_file = os.path.join(config.TABLES_DIR, "analysis_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NUMERICAL METHODS ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Tests Run: {len(df)}\n")
        f.write(f"Successful Convergence: {df['converged'].sum()}/{len(df)}\n")
        f.write(f"Success Rate: {df['converged'].sum()/len(df)*100:.2f}%\n\n")
        
        # Method Performance
        f.write("\nMETHOD PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            f.write(f"\n{method.upper().replace('_', ' ')}:\n")
            f.write(f"  Average Iterations: {method_df['iterations'].mean():.2f}\n")
            f.write(f"  Average Time: {method_df['execution_time_ms'].mean():.4f} ms\n")
            f.write(f"  Success Rate: {method_df['converged'].sum()}/{len(method_df)}\n")
            if 'actual_error' in method_df.columns:
                f.write(f"  Average Error: {method_df['actual_error'].mean():.2e}\n")
        
        # Detailed Results
        f.write("\n\nDETAILED RESULTS BY EQUATION\n")
        f.write("="*80 + "\n")
        for eq_name in df['equation_name'].unique():
            eq_df = df[df['equation_name'] == eq_name]
            f.write(f"\n{eq_name}\n")
            f.write("-"*80 + "\n")
            
            for _, row in eq_df.iterrows():
                f.write(f"\n  Method: {row['method'].upper().replace('_', ' ')}\n")
                f.write(f"    Root: {row.get('root_found', 'N/A')}\n")
                f.write(f"    Iterations: {row.get('iterations', 'N/A')}\n")
                f.write(f"    Error: {row.get('final_error', 'N/A')}\n")
                f.write(f"    Time: {row.get('execution_time_ms', 'N/A')} ms\n")
                f.write(f"    Converged: {'Yes' if row.get('converged', False) else 'No'}\n")
    
    print(f"✓ Saved: {report_file}")
    logging.info(f"Saved text report: {report_file}")
    
    # Step 7: Summary
    print("\n" + "="*80)
    print(" COMPLETE ANALYSIS FINISHED!")
    print("="*80)
    
    print("\n OUTPUT LOCATIONS:")
    print(f"   Tables: {config.TABLES_DIR}/")
    print(f"     • complete_results.csv")
    print(f"     • summary_by_method.csv")
    print(f"     • summary_by_equation.csv")
    print(f"     • iterations_comparison.csv")
    print(f"     • analysis_report.txt")
    
    print(f"\n   Plots: {config.PLOTS_DIR}/")
    print(f"     • convergence_*.png (7 files)")
    print(f"     • function_*.png (7 files)")
    print(f"     • iterations_comparison.png")
    print(f"     • execution_time_comparison.png")
    
    print(f"\n   Logs: {config.LOGS_DIR}/")
    print(f"     • analysis_*.log")
    
    print("\n" + "="*80)
    logging.info("="*80)
    logging.info("Complete analysis finished successfully!")
    logging.info("="*80)
    
    return df


def print_console_summary(df):
    """
    Print a beautiful summary to console.
    """
    from tabulate import tabulate
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Method comparison
    print("\n Method Performance:")
    method_summary = []
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        method_summary.append({
            'Method': method.replace('_', ' ').title(),
            'Avg Iterations': f"{method_df['iterations'].mean():.2f}",
            'Avg Time (ms)': f"{method_df['execution_time_ms'].mean():.4f}",
            'Success Rate': f"{method_df['converged'].sum()}/{len(method_df)}"
        })
    
    print(tabulate(method_summary, headers='keys', tablefmt='grid'))
    
    # Detailed comparison
    print("\n Detailed Comparison:")
    display_df = df[['equation_name', 'method', 'root_found', 'iterations', 'converged']].copy()
    display_df['method'] = display_df['method'].str.replace('_', ' ').str.title()
    display_df['root_found'] = display_df['root_found'].apply(lambda x: f"{x:.8f}" if pd.notna(x) else "N/A")
    display_df['converged'] = display_df['converged'].apply(lambda x: "✓" if x else "✗")
    
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║    
    ║                                                                          ║
    ║                  Estimated time: 1-2 minutes                             ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    input("\nPress ENTER to start the complete analysis...")
    
    # Run everything
    df = generate_all_outputs()
    
    # Print console summary
    print_console_summary(df)
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print("\n Next Steps:")
    print("   1. Check results/tables/ for CSV files")
    print("   2. Check results/plots/ for all images")
    print("   3. Check results/logs/ for execution logs")
    print("   4. Open analysis_report.txt for summary")
    print("\n" + "="*80 + "\n")