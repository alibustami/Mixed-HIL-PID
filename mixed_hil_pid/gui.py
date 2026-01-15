"""
GUI components for human-in-the-loop PID optimization.

Provides two GUI classes:
- MixedHILGUI: For comparing two candidates (DE vs BO)
- SingleHILGUI: For evaluating one candidate (DE-only or BO-only)
"""

import tkinter as tk
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def format_metrics(metrics):
    """Format PID metrics as readable string."""
    settle = f"{metrics['settling_time']:.2f}s" if metrics["settling_time"] > 0 else "Unstable"
    rise = f"{metrics['rise_time']:.2f}s" if metrics["rise_time"] > 0 else "N/A"
    return f"Overshoot: {metrics['overshoot']:.1f}% | Rise: {rise} | Settle: {settle}"


class MixedHILGUI:
    """GUI for Mixed HIL (DE vs BO comparison)."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mixed HIL: DE vs BO")
        self.root.geometry("1100x750")
        self._choice_var = tk.IntVar(value=0)

    def show_comparison(self, hist_a, hist_b, params_a, params_b, metrics_a, metrics_b):
        """Show two candidates and get user preference."""
        self._choice_var.set(0)
        for widget in self.root.winfo_children():
            widget.destroy()

        # Header
        header = tk.Frame(self.root)
        header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Label(header, text=f"A (DE): Kp={params_a[0]:.1f}, Ki={params_a[1]:.1f}, Kd={params_a[2]:.1f}\n{format_metrics(metrics_a)}",
                 font=("Arial", 11, "bold"), bg="#e6f2ff", relief=tk.RAISED, padx=10, pady=5).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Label(header, text=f"B (BO): Kp={params_b[0]:.1f}, Ki={params_b[1]:.1f}, Kd={params_b[2]:.1f}\n{format_metrics(metrics_b)}",
                 font=("Arial", 11, "bold"), bg="#ffe6e6", relief=tk.RAISED, padx=10, pady=5).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Plots
        sns.set_theme(style="whitegrid")
        fig = Figure(figsize=(10, 5), dpi=100)
        
        for idx, (hist, title, color) in enumerate([(hist_a, "DE", "Blues"), (hist_b, "BO", "Reds")]):
            ax = fig.add_subplot(1, 2, idx + 1)
            ax.axhline(y=90, color="k", linestyle="--", alpha=0.5, label="Target")
            ax.plot(hist["time"], hist["actual"], color=sns.color_palette(color, 3)[-1], linewidth=2.4, label="Response")
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Yaw (°)")
            ax.set_ylim(-10, 120)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()

        FigureCanvasTkAgg(fig, master=self.root).get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        btn_style = {"font": ("Arial", 11), "width": 15, "height": 2}
        
        tk.Button(btn_frame, text="Prefer A (DE)", bg="#d1e7dd", command=lambda: self._choice_var.set(1), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="Prefer B (BO)", bg="#f8d7da", command=lambda: self._choice_var.set(2), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="TIE (Refine)", bg="#fff3cd", command=lambda: self._choice_var.set(3), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="REJECT Both", bg="#e2e3e5", command=lambda: self._choice_var.set(4), **btn_style).pack(side=tk.LEFT, padx=30)

        self.root.wait_variable(self._choice_var)
        return int(self._choice_var.get())


class SingleHILGUI:
    """GUI for Single HIL (DE-only or BO-only)."""
    
    def __init__(self, title="Single HIL"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1100x750")
        self._choice_var = tk.IntVar(value=0)

    def show_candidate(self, history, params, fit, metrics, label="Candidate"):
        """Show one candidate and get accept/reject."""
        self._choice_var.set(0)
        for widget in self.root.winfo_children():
            widget.destroy()

        # Header
        header = tk.Frame(self.root)
        header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        info = f"{label}\nKp={params[0]:.2f}, Ki={params[1]:.2f}, Kd={params[2]:.2f}\nFitness: {fit:.4f}\n{format_metrics(metrics)}"
        tk.Label(header, text=info, font=("Arial", 12, "bold"), bg="#e6f2ff", relief=tk.RAISED, padx=10, pady=10).pack(fill=tk.X, padx=5)

        # Plot
        sns.set_theme(style="whitegrid")
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.axhline(y=90, color="k", linestyle="--", alpha=0.5, label="Target")
        ax.plot(history["time"], history["actual"], color="blue", linewidth=2.5, label="Response")
        ax.set_title(f"Response: {label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Yaw (°)")
        ax.set_ylim(-10, 120)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        FigureCanvasTkAgg(fig, master=self.root).get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        btn_style = {"font": ("Arial", 12, "bold"), "width": 20, "height": 2}
        
        tk.Button(btn_frame, text="ACCEPT (Refine)", bg="#d1e7dd", fg="#0f5132", command=lambda: self._choice_var.set(1), **btn_style).pack(side=tk.LEFT, padx=50, expand=True)
        tk.Button(btn_frame, text="REJECT (Expand)", bg="#f8d7da", fg="#842029", command=lambda: self._choice_var.set(2), **btn_style).pack(side=tk.RIGHT, padx=50, expand=True)

        self.root.wait_variable(self._choice_var)
        return int(self._choice_var.get())
