"""
Visualization Module.
Functions for creating plots and visualizations for the report.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict
import os

import config
from methods import bisection, newton_raphson, secant
from test_functions import get_test_equations


# Set matplotlib style
try:
    plt.style.use(config.PLOT_STYLE)
except:
    plt.style.use('default')


def plot_convergence_history(eq, methods_to_compare=None, save=True):
    """
    Plot convergence history for all methods on a single equation.
    
    Parameters:
    -----------
    eq : TestEquation
        Test equation object
    methods_to_compare : list, optional
        List of method names to compare
    save : bool
        Whether to save the plot
    """
    
    if methods_to_compare is None:
        methods_to_compare = ['bisection', 'newton_raphson', 'secant']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Run methods and collect history
    results = {}
    
    if 'bisection' in methods_to_compare:
        a, b = eq.bisection_interval
        results['bisection'] = bisection(
            eq.f, a, b, 
            tol=config.DEFAULT_TOLERANCE, 
            max_iter=config.MAX_ITERATIONS
        )
    
    if 'newton_raphson' in methods_to_compare:
        results['newton_raphson'] = newton_raphson(
            eq.f, eq.df, eq.newton_guess,
            tol=config.DEFAULT_TOLERANCE,
            max_iter=config.MAX_ITERATIONS
        )
    
    if 'secant' in methods_to_compare:
        x0, x1 = eq.secant_guesses
        results['secant'] = secant(
            eq.f, x0, x1,
            tol=config.DEFAULT_TOLERANCE,
            max_iter=config.MAX_ITERATIONS
        )
    
    # Plot 1: Root approximation history
    ax1 = axes[0]
    for method_name, result in results.items():
        if result['converged']:
            iterations = range(len(result['history']))
            ax1.plot(
                iterations, result['history'],
                marker=config.METHOD_MARKERS[method_name],
                linestyle=config.METHOD_LINESTYLES[method_name],
                color=config.METHOD_COLORS[method_name],
                label=method_name.replace('_', ' ').title(),
                linewidth=2,
                markersize=6
            )
    
    # Add true root line
    if eq.known_roots:
        ax1.axhline(
            y=eq.known_roots[0], 
            color='black', 
            linestyle=':', 
            linewidth=2,
            label='True Root'
        )
    
    ax1.set_xlabel('Iteration', fontsize=config.LABEL_FONTSIZE)
    ax1.set_ylabel('Root Approximation', fontsize=config.LABEL_FONTSIZE)
    ax1.set_title(f'Root Approximation History\n{eq.name}', 
                  fontsize=config.TITLE_FONTSIZE)
    ax1.legend(fontsize=config.LEGEND_FONTSIZE)
    ax1.grid(True, alpha=config.GRID_ALPHA, linestyle=config.GRID_LINESTYLE)
    
    # Plot 2: Error evolution
    ax2 = axes[1]
    for method_name, result in results.items():
        if result['converged'] and result['error_history']:
            iterations = range(1, len(result['error_history']) + 1)
            ax2.semilogy(
                iterations, result['error_history'],
                marker=config.METHOD_MARKERS[method_name],
                linestyle=config.METHOD_LINESTYLES[method_name],
                color=config.METHOD_COLORS[method_name],
                label=method_name.replace('_', ' ').title(),
                linewidth=2,
                markersize=6
            )
    
    ax2.set_xlabel('Iteration', fontsize=config.LABEL_FONTSIZE)
    ax2.set_ylabel('Error (log scale)', fontsize=config.LABEL_FONTSIZE)
    ax2.set_title(f'Error Evolution\n{eq.name}', 
                  fontsize=config.TITLE_FONTSIZE)
    ax2.legend(fontsize=config.LEGEND_FONTSIZE)
    ax2.grid(True, alpha=config.GRID_ALPHA, linestyle=config.GRID_LINESTYLE)
    
    plt.tight_layout()
    
    if save:
        filename = f"{config.PLOTS_DIR}/convergence_{eq.name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f" Saved: {filename}")
    
    plt.show()
    
    return fig


def plot_iterations_comparison(results_df, save=True):
    """
    Create bar chart comparing iterations for all methods across equations.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from main.py
    save : bool
        Whether to save the plot
    """
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    # Prepare data
    equations = results_df['equation_name'].unique()
    methods = results_df['method'].unique()
    
    x = np.arange(len(equations))
    width = 0.25
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        iterations = [
            method_data[method_data['equation_name'] == eq]['iterations'].values[0]
            if len(method_data[method_data['equation_name'] == eq]) > 0 else 0
            for eq in equations
        ]
        
        ax.bar(
            x + i * width, iterations, width,
            label=method.replace('_', ' ').title(),
            color=config.METHOD_COLORS[method],
            alpha=0.8
        )
    
    ax.set_xlabel('Test Equation', fontsize=config.LABEL_FONTSIZE)
    ax.set_ylabel('Number of Iterations', fontsize=config.LABEL_FONTSIZE)
    ax.set_title('Iterations Comparison Across Methods', 
                 fontsize=config.TITLE_FONTSIZE)
    ax.set_xticks(x + width)
    ax.set_xticklabels(equations, rotation=45, ha='right')
    ax.legend(fontsize=config.LEGEND_FONTSIZE)
    ax.grid(True, axis='y', alpha=config.GRID_ALPHA, linestyle=config.GRID_LINESTYLE)
    
    plt.tight_layout()
    
    if save:
        filename = f"{config.PLOTS_DIR}/{config.ITERATIONS_PLOT_NAME}"
        plt.savefig(filename, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f" Saved: {filename}")
    
    plt.show()
    
    return fig


def plot_execution_time_comparison(results_df, save=True):
    """
    Create bar chart comparing execution times.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    save : bool
        Whether to save the plot
    """
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    equations = results_df['equation_name'].unique()
    methods = results_df['method'].unique()
    
    x = np.arange(len(equations))
    width = 0.25
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        times = [
            method_data[method_data['equation_name'] == eq]['execution_time'].values[0] * 1000
            if len(method_data[method_data['equation_name'] == eq]) > 0 else 0
            for eq in equations
        ]
        
        ax.bar(
            x + i * width, times, width,
            label=method.replace('_', ' ').title(),
            color=config.METHOD_COLORS[method],
            alpha=0.8
        )
    
    ax.set_xlabel('Test Equation', fontsize=config.LABEL_FONTSIZE)
    ax.set_ylabel('Execution Time (ms)', fontsize=config.LABEL_FONTSIZE)
    ax.set_title('Execution Time Comparison', 
                 fontsize=config.TITLE_FONTSIZE)
    ax.set_xticks(x + width)
    ax.set_xticklabels(equations, rotation=45, ha='right')
    ax.legend(fontsize=config.LEGEND_FONTSIZE)
    ax.grid(True, axis='y', alpha=config.GRID_ALPHA, linestyle=config.GRID_LINESTYLE)
    
    plt.tight_layout()
    
    if save:
        filename = f"{config.PLOTS_DIR}/execution_time_comparison.png"
        plt.savefig(filename, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f" Saved: {filename}")
    
    plt.show()
    
    return fig


def plot_function_with_roots(eq, save=True):
    """
    Plot the function and mark the roots found by each method.
    
    Parameters:
    -----------
    eq : TestEquation
        Test equation
    save : bool
        Whether to save the plot
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create x range
    a, b = eq.bisection_interval
    margin = (b - a) * 0.3
    x = np.linspace(a - margin, b + margin, 1000)
    y = np.array([eq.f(xi) for xi in x])
    
    # Plot function
    ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Run methods and mark roots
    results = {}
    
    # Bisection
    results['bisection'] = bisection(eq.f, a, b, tol=config.DEFAULT_TOLERANCE)
    
    # Newton-Raphson
    results['newton_raphson'] = newton_raphson(
        eq.f, eq.df, eq.newton_guess, tol=config.DEFAULT_TOLERANCE
    )
    
    # Secant
    x0, x1 = eq.secant_guesses
    results['secant'] = secant(eq.f, x0, x1, tol=config.DEFAULT_TOLERANCE)
    
    # Mark roots
    for method_name, result in results.items():
        if result['converged']:
            root = result['root']
            ax.plot(
                root, 0,
                marker=config.METHOD_MARKERS[method_name],
                color=config.METHOD_COLORS[method_name],
                markersize=12,
                label=f"{method_name.replace('_', ' ').title()}: x={root:.6f}"
            )
    
    # Mark known root
    if eq.known_roots:
        ax.plot(
            eq.known_roots[0], 0,
            'k*', markersize=15,
            label=f"True Root: x={eq.known_roots[0]:.6f}"
        )
    
    ax.set_xlabel('x', fontsize=config.LABEL_FONTSIZE)
    ax.set_ylabel('f(x)', fontsize=config.LABEL_FONTSIZE)
    ax.set_title(f'Function Plot: {eq.name}\n{eq.description}', 
                 fontsize=config.TITLE_FONTSIZE)
    ax.legend(fontsize=config.LEGEND_FONTSIZE)
    ax.grid(True, alpha=config.GRID_ALPHA, linestyle=config.GRID_LINESTYLE)
    
    plt.tight_layout()
    
    if save:
        filename = f"{config.PLOTS_DIR}/function_{eq.name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f" Saved: {filename}")
    
    plt.show()
    
    return fig


def generate_all_plots(results_df=None):
    """
    Generate all plots for the report.
    
    Parameters:
    -----------
    results_df : pd.DataFrame, optional
        Results dataframe. If None, will be generated.
    """
    
    print("\n" + "="*80)
    print("GENERATING ALL PLOTS")
    print("="*80)
    
    equations = get_test_equations()
    
    # Generate convergence plots for each equation
    print("\n Generating convergence history plots..")
    for eq_id, eq in equations.items():
        print(f"  Processing: {eq.name}")
        plot_convergence_history(eq, save=True)
        plt.close()
    
    # Generate function plots
    print("\n Generating function plots..")
    for eq_id, eq in equations.items():
        print(f"  Processing: {eq.name}")
        plot_function_with_roots(eq, save=True)
        plt.close()
    
    # Generate comparison plots if results_df provided
    if results_df is not None:
        print("\n Generating comparison plots..")
        plot_iterations_comparison(results_df, save=True)
        plt.close()
        
        plot_execution_time_comparison(results_df, save=True)
        plt.close()
    
    print("\n" + "="*80)
    print(" ALL PLOTS GENERATED")
    print(f" Plots saved in: {config.PLOTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example: Generate all plots
    equations = get_test_equations()
    
    # Generate plots for first equation as example
    eq = equations['eq1, eq2, eq3']  
    
    print("Generating example plots...")
    plot_convergence_history(eq)
    plot_function_with_roots(eq)
    
    print("\n To generate all plots, run:")
    print("   python -c 'from analysis.visualizations import generate_all_plots; generate_all_plots()'")