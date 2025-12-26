import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class FeedbackGUI:
    def __init__(self):
        self.user_choice = None
        self.root = tk.Tk()
        self.root.title("PID Tuning Dashboard")
        self.root.geometry("1100x750")

    def show_comparison(self, history_a, history_b, params_a, params_b, fit_a, fit_b, metrics_a, metrics_b):
        self.user_choice = None 
        
        for widget in self.root.winfo_children():
            widget.destroy()

        # --- INFO HEADER ---
        header_frame = tk.Frame(self.root)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Helper to format metrics text
        def format_metrics(m):
            settle = f"{m['settling_time']:.2f}s" if m['settling_time'] > 0 else "Unstable"
            rise = f"{m['rise_time']:.2f}s" if m['rise_time'] > 0 else "N/A"
            return f"Overshoot: {m['overshoot']:.1f}% | Rise Time: {rise} | Settling Time: {settle}"

        # DE Info
        lbl_a = tk.Label(header_frame, 
                         text=f"A (DE): Kp={params_a[0]:.1f}, Ki={params_a[1]:.1f}, Kd={params_a[2]:.1f}\n{format_metrics(metrics_a)}",
                         font=("Arial", 11, "bold"), bg="#e6f2ff", relief=tk.RAISED, padx=10, pady=5)
        lbl_a.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # BO Info
        lbl_b = tk.Label(header_frame, 
                         text=f"B (BO): Kp={params_b[0]:.1f}, Ki={params_b[1]:.1f}, Kd={params_b[2]:.1f}\n{format_metrics(metrics_b)}",
                         font=("Arial", 11, "bold"), bg="#ffe6e6", relief=tk.RAISED, padx=10, pady=5)
        lbl_b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # --- PLOTS ---
        fig = Figure(figsize=(10, 5), dpi=100)
        
        for i, (hist, title, color) in enumerate([
            (history_a, "Candidate A (DE)", 'b'), 
            (history_b, "Candidate B (BO)", 'r')
        ]):
            ax = fig.add_subplot(1, 2, i+1)
            ax.axhline(y=90, color='k', linestyle='--', alpha=0.5, label='Target')
            ax.plot(hist['time'], hist['actual'], color=color, linewidth=2, label='Response')
            
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Yaw (Deg)")
            ax.set_ylim(-10, 120)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- BUTTONS ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        btn_style = {"font": ("Arial", 11), "width": 15, "height": 2}

        tk.Button(btn_frame, text="Prefer A (DE)", bg="#d1e7dd", command=lambda: self._set_choice(1), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="Prefer B (BO)", bg="#f8d7da", command=lambda: self._set_choice(2), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="TIE (Refine)", bg="#fff3cd", command=lambda: self._set_choice(3), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="REJECT Both", bg="#e2e3e5", command=lambda: self._set_choice(4), **btn_style).pack(side=tk.LEFT, padx=30)

        self.root.mainloop()
        return self.user_choice

    def _set_choice(self, choice):
        self.user_choice = choice
        self.root.quit()