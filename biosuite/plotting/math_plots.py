"""Mathematical functions: Sine, Cosine, Linear, Quadratic, Cubic, Exponential, Logistic."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..core.utils import config, session, autosave_session, safe_float_input, apply_glass_ax, ask_save_plot

def sine_plot(pdf=None):
    print("\n--- Sine: y = A * sin(B*x + C) ---")
    try:
        A = safe_float_input("Amplitude A (default 1): ", 1, key='sine_A')
        B = safe_float_input("Frequency B (default 1): ", 1, key='sine_B')
        C = safe_float_input("Phase C (default 0): ", 0, key='sine_C')
        x = np.linspace(0, 2*np.pi, 300)
        y = A * np.sin(B*x + C)
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='blue', linewidth=2, ax=ax)
        ax.set_title(f'Sine: y = {A}·sin({B}x + {C})')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('sine', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def cosine_plot(pdf=None):
    print("\n--- Cosine: y = A * cos(B*x + C) ---")
    try:
        A = safe_float_input("Amplitude A (default 1): ", 1, key='cosine_A')
        B = safe_float_input("Frequency B (default 1): ", 1, key='cosine_B')
        C = safe_float_input("Phase C (default 0): ", 0, key='cosine_C')
        x = np.linspace(0, 2*np.pi, 300)
        y = A * np.cos(B*x + C)
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='green', linewidth=2, ax=ax)
        ax.set_title(f'Cosine: y = {A}·cos({B}x + {C})')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('cosine', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def linear_plot(pdf=None):
    print("\n--- Linear: y = a*x + b ---")
    try:
        a = safe_float_input("Slope a (default 2): ", 2, key='linear_a')
        b = safe_float_input("Intercept b (default 1): ", 1, key='linear_b')
        x = np.linspace(-5, 5, 100)
        y = a*x + b
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='purple', linewidth=2, ax=ax)
        ax.set_title(f'Linear: y = {a}x + {b}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('linear', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def quadratic_plot(pdf=None):
    print("\n--- Quadratic: y = a*x² + b*x + c ---")
    try:
        a = safe_float_input("a (default 1): ", 1, key='quad_a')
        b = safe_float_input("b (default -3): ", -3, key='quad_b')
        c = safe_float_input("c (default 2): ", 2, key='quad_c')
        x = np.linspace(-5, 5, 100)
        y = a*x**2 + b*x + c
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='red', linewidth=2, ax=ax)
        ax.set_title(f'Quadratic: y = {a}x² + {b}x + {c}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('quadratic', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def cubic_plot(pdf=None):
    print("\n--- Cubic: y = a*x³ + b*x² + c*x + d ---")
    try:
        a = safe_float_input("a (default 1): ", 1, key='cubic_a')
        b = safe_float_input("b (default -2): ", -2, key='cubic_b')
        c = safe_float_input("c (default 1): ", 1, key='cubic_c')
        d = safe_float_input("d (default 0): ", 0, key='cubic_d')
        x = np.linspace(-4, 4, 100)
        y = a*x**3 + b*x**2 + c*x + d
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='orange', linewidth=2, ax=ax)
        ax.set_title(f'Cubic: y = {a}x³ + {b}x² + {c}x + {d}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('cubic', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def exponential_plot(pdf=None):
    print("\n--- Exponential: y = a * exp(b*x) + c ---")
    try:
        a = safe_float_input("Scale a (default 1): ", 1, key='exp_a')
        b = safe_float_input("Rate b (default 0.5): ", 0.5, key='exp_b')
        c = safe_float_input("Offset c (default 0): ", 0, key='exp_c')
        x = np.linspace(0, 5, 100)
        y = a * np.exp(b*x) + c
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='teal', linewidth=2, ax=ax)
        ax.set_title(f'Exponential: y = {a}·exp({b}x) + {c}')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('exponential', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")

def logistic_plot(pdf=None):
    print("\n--- Logistic: y = L / (1 + exp(-k*(x - x0))) ---")
    try:
        L = safe_float_input("Carrying capacity L (default 1): ", 1, key='log_L')
        k = safe_float_input("Growth rate k (default 1): ", 1, key='log_k')
        x0 = safe_float_input("Midpoint x0 (default 0): ", 0, key='log_x0')
        x = np.linspace(-6, 6, 200)
        y = L / (1 + np.exp(-k*(x - x0)))
        fig, ax = plt.subplots(figsize=(7,4))
        sns.lineplot(x=x, y=y, color='darkred', linewidth=2, ax=ax)
        ax.set_title(f'Logistic: y = {L}/(1+exp(-{k}(x-{x0})))')
        ax.grid(True, alpha=0.3)
        apply_glass_ax(ax)
        ask_save_plot('logistic', config['save_format'], config['default_dpi'], pdf)
        plt.close('all')
        plt.close()
    except Exception as e:
        print(f"Error: {e}")