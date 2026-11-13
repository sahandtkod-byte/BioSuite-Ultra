"""Mathematical functions: Sine, Cosine, Linear, Quadratic, Cubic, Exponential, Logistic."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..core.utils import (
    apply_glass_ax,
    ask_save_plot,
    config,
    safe_float_input,
)
from ._demo_guard import demo_guard as _demo


# Public surface. ``biosuite/plotting/__init__.py`` star-imports this
# module, and without this list it re-exports the module's own imports
# (numpy, pandas, os, plotly, ...) as package-level names too.
__all__ = [
    'sine_plot',
    'cosine_plot',
    'linear_plot',
    'quadratic_plot',
    'cubic_plot',
    'exponential_plot',
    'logistic_plot',
]


def _finite_curve(y, expression):
    """Refuse a curve with infinities instead of drawing a plot of them."""
    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"{expression} is not finite for these parameters; "
            "use smaller coefficients or a narrower range")


def _stable_logistic(L, k, x, x0):
    """Evaluate L / (1 + exp(-k(x - x0))) without overflowing exp()."""
    z = np.clip(k * (x - x0), -700, 700)
    return L / (1 + np.exp(-z))


def sine_plot(pdf=None):
    print("\n--- Sine: y = A * sin(B*x + C) ---")
    with _demo("sine_plot"):
        A = safe_float_input("Amplitude A (default 1): ", 1, key='sine_A')
        B = safe_float_input("Frequency B (default 1): ", 1, key='sine_B')
        C = safe_float_input("Phase C (default 0): ", 0, key='sine_C')
        x = np.linspace(0, 2*np.pi, 300)
        y = A * np.sin(B*x + C)
        _finite_curve(y, "the sine curve")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='blue', linewidth=2, ax=ax)
        ax.set_title(f'Sine: y = {A}·sin({B}x + {C})')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('sine', config['save_format'], config['default_dpi'], pdf)


def cosine_plot(pdf=None):
    print("\n--- Cosine: y = A * cos(B*x + C) ---")
    with _demo("cosine_plot"):
        A = safe_float_input("Amplitude A (default 1): ", 1, key='cosine_A')
        B = safe_float_input("Frequency B (default 1): ", 1, key='cosine_B')
        C = safe_float_input("Phase C (default 0): ", 0, key='cosine_C')
        x = np.linspace(0, 2*np.pi, 300)
        y = A * np.cos(B*x + C)
        _finite_curve(y, "the cosine curve")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='green', linewidth=2, ax=ax)
        ax.set_title(f'Cosine: y = {A}·cos({B}x + {C})')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('cosine', config['save_format'], config['default_dpi'], pdf)


def linear_plot(pdf=None):
    print("\n--- Linear: y = a*x + b ---")
    with _demo("linear_plot"):
        a = safe_float_input("Slope a (default 2): ", 2, key='linear_a')
        b = safe_float_input("Intercept b (default 1): ", 1, key='linear_b')
        x = np.linspace(-5, 5, 100)
        y = a*x + b
        _finite_curve(y, "the linear curve")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='purple', linewidth=2, ax=ax)
        ax.set_title(f'Linear: y = {a}x + {b}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('linear', config['save_format'], config['default_dpi'], pdf)


def quadratic_plot(pdf=None):
    print("\n--- Quadratic: y = a*x² + b*x + c ---")
    with _demo("quadratic_plot"):
        a = safe_float_input("a (default 1): ", 1, key='quad_a')
        b = safe_float_input("b (default -3): ", -3, key='quad_b')
        c = safe_float_input("c (default 2): ", 2, key='quad_c')
        x = np.linspace(-5, 5, 100)
        y = a*x**2 + b*x + c
        _finite_curve(y, "the quadratic curve")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='red', linewidth=2, ax=ax)
        ax.set_title(f'Quadratic: y = {a}x² + {b}x + {c}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('quadratic', config['save_format'], config['default_dpi'], pdf)


def cubic_plot(pdf=None):
    print("\n--- Cubic: y = a*x³ + b*x² + c*x + d ---")
    with _demo("cubic_plot"):
        a = safe_float_input("a (default 1): ", 1, key='cubic_a')
        b = safe_float_input("b (default -2): ", -2, key='cubic_b')
        c = safe_float_input("c (default 1): ", 1, key='cubic_c')
        d = safe_float_input("d (default 0): ", 0, key='cubic_d')
        x = np.linspace(-4, 4, 100)
        y = a*x**3 + b*x**2 + c*x + d
        _finite_curve(y, "the cubic curve")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='orange', linewidth=2, ax=ax)
        ax.set_title(f'Cubic: y = {a}x³ + {b}x² + {c}x + {d}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('cubic', config['save_format'], config['default_dpi'], pdf)


def exponential_plot(pdf=None):
    print("\n--- Exponential: y = a * exp(b*x) + c ---")
    with _demo("exponential_plot"):
        a = safe_float_input("Scale a (default 1): ", 1, key='exp_a')
        b = safe_float_input("Rate b (default 0.5): ", 0.5, key='exp_b')
        c = safe_float_input("Offset c (default 0): ", 0, key='exp_c')
        x = np.linspace(0, 5, 100)
        y = a * np.exp(b*x) + c
        _finite_curve(y, "the exponential curve")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='teal', linewidth=2, ax=ax)
        ax.set_title(f'Exponential: y = {a}·exp({b}x) + {c}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('exponential', config['save_format'], config['default_dpi'], pdf)


def logistic_plot(pdf=None):
    print("\n--- Logistic: y = L / (1 + exp(-k*(x - x0))) ---")
    with _demo("logistic_plot"):
        L = safe_float_input("Carrying capacity L (default 1): ", 1, key='log_L')
        k = safe_float_input("Growth rate k (default 1): ", 1, key='log_k')
        x0 = safe_float_input("Midpoint x0 (default 0): ", 0, key='log_x0')
        x = np.linspace(-6, 6, 200)
        y = _stable_logistic(L, k, x, x0)
        _finite_curve(y, "the logistic curve")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='darkred', linewidth=2, ax=ax)
        ax.set_title(f'Logistic: y = {L}/(1+exp(-{k}(x-{x0})))')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('logistic', config['save_format'], config['default_dpi'], pdf)
