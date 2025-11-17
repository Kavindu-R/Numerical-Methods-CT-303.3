
import os

# PROJECT METADATA

PROJECT_NAME = "Numerical Methods for Solving Nonlinear Equations"
COURSE_CODE = "CS303.3 -CT"

# ALGORITHM SETTINGS

# Default convergence settings
DEFAULT_TOLERANCE = 1e-6
MAX_ITERATIONS = 100

# Bisection method specific
BISECTION_MAX_ITER = 100

# Newton-Raphson specific
NEWTON_MAX_ITER = 100
NEWTON_DERIVATIVE_THRESHOLD = 1e-12

# Secant method specific
SECANT_MAX_ITER = 100

# TEST EQUATION SETTINGS

# Which equations to test (set to None to test all)
EQUATIONS_TO_TEST = None  # ['eq1', 'eq2', 'eq3', 'eq4', 'eq5', 'eq6', 'eq7']

# Test all equations if None
TEST_ALL_EQUATIONS = True

# OUTPUT SETTINGS

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Create directories if they don't exist
for directory in [RESULTS_DIR, TABLES_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File naming conventions
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
RESULTS_CSV_NAME = "numerical_methods_results.csv"
CONVERGENCE_PLOT_NAME = "convergence_comparison.png"
ERROR_PLOT_NAME = "error_evolution.png"
ITERATIONS_PLOT_NAME = "iterations_comparison.png"

# PLOTTING SETTINGS

# Matplotlib settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Try different styles if this doesn't work
FIGURE_SIZE = (12, 8)
PLOT_DPI = 300
SAVE_FORMAT = 'png'  # 'png', 'pdf', 'svg'

# Color scheme for methods
METHOD_COLORS = {
    'bisection': '#e74c3c',      # Red
    'newton_raphson': '#3498db',  # Blue
    'secant': '#2ecc71'           # Green
}

# Line styles
METHOD_LINESTYLES = {
    'bisection': '-',
    'newton_raphson': '--',
    'secant': '-.'
}

# Markers
METHOD_MARKERS = {
    'bisection': 'o',
    'newton_raphson': 's',
    'secant': '^'
}

# Font sizes
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
TICK_FONTSIZE = 10

# Grid settings
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'

# COMPARISON SETTINGS

# Metrics to compare
COMPARE_METRICS = [
    'iterations',
    'final_error',
    'function_value',
    'execution_time',
    'function_evaluations'
]

# Display precision
DISPLAY_PRECISION = 10
ERROR_PRECISION = 2  # Scientific notation

# LOGGING SETTINGS

# Verbosity levels
VERBOSE = True
DEBUG = False

# Log file settings
LOG_FILE = os.path.join(LOGS_DIR, "execution.log")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# PERFORMANCE SETTINGS

# Number of runs for timing (average over multiple runs)
TIMING_RUNS = 5

# Whether to track detailed history
TRACK_HISTORY = True

# REPORT SETTINGS

# Report generation
GENERATE_LATEX = False  # Set to True if you want LaTeX tables
GENERATE_MARKDOWN = True

# Table format
TABLE_FORMAT = 'grid'  # 'grid', 'fancy_grid', 'pipe', 'html', 'latex'

# VALIDATION SETTINGS

# Root validation tolerance
ROOT_VALIDATION_TOLERANCE = 1e-6

# Check convergence
CHECK_CONVERGENCE = True

# Warn on non-convergence
WARN_NON_CONVERGENCE = True

# FUNCTIONS

def print_config():
    """Print current configuration."""
    print("\n" + "="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    print(f"Project: {PROJECT_NAME}")
    print(f"Course: {COURSE_CODE}")
    print()
    print(f"Tolerance: {DEFAULT_TOLERANCE}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print()
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Plots Directory: {PLOTS_DIR}")
    print(f"Tables Directory: {TABLES_DIR}")
    print()
    print(f"Plot Style: {PLOT_STYLE}")
    print(f"Figure Size: {FIGURE_SIZE}")
    print(f"DPI: {PLOT_DPI}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_config()