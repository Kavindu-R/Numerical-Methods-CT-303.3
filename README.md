**CS303.3 -CT-22.2/23.1 Group Coursework**  
# Numerical Methods for Solving Nonlinear Equations

## 📋 Project Overview

This project implements and compares three numerical methods for solving nonlinear equations:
- **Bisection Method** - Reliable, guaranteed convergence (linear)
- **Newton-Raphson Method** - Fast, quadratic convergence
- **Secant Method** - No derivative needed, super-linear convergence

## 🎯 Learning Objectives

✅ Understand theoretical background of numerical methods  
✅ Implement algorithms in Python  
✅ Analyze convergence properties  
✅ Compare computational complexity and accuracy  
✅ Solve practical problems using numerical methods

## 📁 Project Structure

```
numerical_methods_project/
│
├── methods/                    # Core implementations
│   ├── bisection.py           # Bisection method
│   ├── newton_raphson.py      # Newton-Raphson method
│   └── secant.py              # Secant method
│
├── test_functions/            # Test equations
│   └── equations.py           # 7 test equations with known roots
│
├── analysis/                  # Analysis and visualization
│   └── visualizations.py      # Plotting functions
│
├── results/                   # Generated outputs
│   ├── tables/               # CSV results
│   ├── plots/                # PNG figures
│   └── logs/                 # Execution logs
│
├── config.py                  # Configuration settings
├── main.py                    # Main execution script
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Main Analysis

```bash
python main.py
```

This will:
- Run all three methods on all test equations
- Generate comparison tables
- Save results to CSV
- Display summary statistics

### 3. Generate Plots

```bash
# Generate all visualization plots
python -c "from analysis.visualizations import generate_all_plots; generate_all_plots()"
```

### 4. Test Individual Methods

```bash
# Test bisection method
python methods/bisection.py

# Test Newton-Raphson method
python methods/newton_raphson.py

# Test secant method
python methods/secant.py
```

## 📊 Test Equations

The project includes 7 carefully selected test equations:

| ID | Equation | Type | Known Root |
|----|----------|------|------------|
| eq1 | x² - 4 = 0 | Polynomial | 2.0 |
| eq2 | x³ - x - 2 = 0 | Cubic | 1.5213797... |
| eq3 | cos(x) - x = 0 | Transcendental | 0.7390851... |
| eq4 | e^x - 3x = 0 | Exponential | 1.5121345... |
| eq5 | x³ - 2x - 5 = 0 | Cubic | 2.0945514... |
| eq6 | x·sin(x) - 1 = 0 | Trig Product | 1.1141571... |
| eq7 | x² - e^(-x) = 0 | Mixed | 0.7034674... |

## 📈 Key Features

### Implementation Features
- ✅ Complete error handling and validation
- ✅ Detailed iteration tracking
- ✅ Verbose modes for debugging
- ✅ Automatic derivative calculation (Newton-Raphson)
- ✅ Convergence criteria based on tolerance
- ✅ Maximum iteration limits

### Analysis Features
- ✅ Convergence speed comparison
- ✅ Accuracy analysis
- ✅ Execution time measurement
- ✅ Function evaluation counting
- ✅ Automated visualization generation

### Visualization Features
- ✅ Convergence history plots
- ✅ Error evolution (log scale)
- ✅ Function plots with roots marked
- ✅ Iterations comparison bar charts
- ✅ Execution time comparison

## 📝 Usage Examples

### Example 1: Solve a Single Equation

```python
from methods import bisection
import numpy as np

# Define function
def f(x):
    return x**2 - 4

# Solve using bisection
result = bisection(f, 0, 3, tol=1e-6)

print(f"Root: {result['root']}")
print(f"Iterations: {result['iterations']}")
print(f"Converged: {result['converged']}")
```

### Example 2: Compare All Methods

```python
from methods import bisection, newton_raphson, secant
from test_functions import get_equation_by_id

# Get test equation
eq = get_equation_by_id('eq3')  # cos(x) - x = 0

# Run all methods
a, b = eq.bisection_interval
r1 = bisection(eq.f, a, b)

r2 = newton_raphson(eq.f, eq.df, eq.newton_guess)

x0, x1 = eq.secant_guesses
r3 = secant(eq.f, x0, x1)

# Compare
print(f"Bisection: {r1['iterations']} iterations")
print(f"Newton-Raphson: {r2['iterations']} iterations")
print(f"Secant: {r3['iterations']} iterations")
```

### Example 3: Generate Plots

```python
from analysis.visualizations import plot_convergence_history
from test_functions import get_equation_by_id

eq = get_equation_by_id('eq2')
plot_convergence_history(eq, save=True)
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Convergence settings
DEFAULT_TOLERANCE = 1e-6
MAX_ITERATIONS = 100

# Plot settings
PLOT_DPI = 300
FIGURE_SIZE = (12, 8)

# Output directories
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
```

## 📊 Expected Results

### Convergence Speed (Typical)
- **Bisection:** 15-25 iterations
- **Newton-Raphson:** 3-6 iterations
- **Secant:** 5-8 iterations

### Convergence Order
- **Bisection:** Linear (order 1)
- **Newton-Raphson:** Quadratic (order 2)
- **Secant:** Super-linear (order φ ≈ 1.618)

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Make sure you're in the project root directory and virtual environment is activated.

### Issue: Plots not displaying
**Solution:** 
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

### Issue: Method fails to converge
**Solution:** Check initial guesses/intervals. Some equations may need different starting points.

## 📚 Dependencies

- `numpy >= 1.24.0` - Numerical computations
- `matplotlib >= 3.7.0` - Plotting
- `pandas >= 2.0.0` - Data organization
- `scipy >= 1.10.0` - Validation (optional)
- `tabulate >= 0.9.0` - Pretty tables

## 👥 Contributors



## 📖 References

1. Burden, R. L., & Faires, J. D. (2010). *Numerical Analysis* (9th ed.)
2. Chapra, S. C., & Canale, R. P. (2015). *Numerical Methods for Engineers*
3. Heath, M. T. (2018). *Scientific Computing: An Introductory Survey*


## ⚖️ License

This project is created for educational purposes as part of CS303.3 coursework.

---
*Last updated: November 2025*