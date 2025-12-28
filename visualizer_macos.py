import tkinter as tk
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class FeedbackGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PID Tuning Dashboard")
        self.root.geometry("1100x750")

        self._choice_var = tk.IntVar(value=0)  # 0 = none yet

    def show_comparison(self, history_a, history_b, params_a, params_b, fit_a, fit_b, metrics_a, metrics_b, prev_de_histories, prev_bo_histories):
        self._choice_var.set(0)

        for widget in self.root.winfo_children():
            widget.destroy()

        header_frame = tk.Frame(self.root)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        def format_metrics(m):
            settle = f"{m['settling_time']:.2f}s" if m["settling_time"] > 0 else "Unstable"
            rise = f"{m['rise_time']:.2f}s" if m["rise_time"] > 0 else "N/A"
            return f"Overshoot: {m['overshoot']:.1f}% | Rise Time: {rise} | Settling Time: {settle}"

        lbl_a = tk.Label(
            header_frame,
            text=f"A (DE): Kp={params_a[0]:.1f}, Ki={params_a[1]:.1f}, Kd={params_a[2]:.1f}\n{format_metrics(metrics_a)}",
            font=("Arial", 11, "bold"), bg="#e6f2ff", relief=tk.RAISED, padx=10, pady=5
        )
        lbl_a.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        lbl_b = tk.Label(
            header_frame,
            text=f"B (BO): Kp={params_b[0]:.1f}, Ki={params_b[1]:.1f}, Kd={params_b[2]:.1f}\n{format_metrics(metrics_b)}",
            font=("Arial", 11, "bold"), bg="#ffe6e6", relief=tk.RAISED, padx=10, pady=5
        )
        lbl_b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        sns.set_theme(style="whitegrid")
        fig = Figure(figsize=(10, 5), dpi=100)

        def plot_family(ax, current_hist, prev_hists, title, palette_name):
            n_prev = len(prev_hists)
            colors = sns.color_palette(palette_name, n_colors=max(n_prev + 2, 3))
            ax.axhline(y=90, color="k", linestyle="--", alpha=0.5, label="Target")

            # Older responses: fader colors
            for idx, hist in enumerate(prev_hists):
                alpha = 0.2 + 0.6 * (idx + 1) / max(n_prev, 1)
                ax.plot(hist["time"], hist["actual"], color=colors[idx], linewidth=1.2, alpha=alpha, label="_prev")

            # Current response
            ax.plot(current_hist["time"], current_hist["actual"], color=colors[-1], linewidth=2.4, label="Current")
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Yaw (Deg)")
            ax.set_ylim(-10, 120)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()

        ax_de = fig.add_subplot(1, 2, 1)
        plot_family(ax_de, history_a, prev_de_histories, "Candidate A (DE)", "Blues")

        ax_bo = fig.add_subplot(1, 2, 2)
        plot_family(ax_bo, history_b, prev_bo_histories, "Candidate B (BO)", "Reds")

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        btn_style = {"font": ("Arial", 11), "width": 15, "height": 2}

        tk.Button(btn_frame, text="Prefer A (DE)", bg="#d1e7dd",
                  command=lambda: self._choice_var.set(1), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="Prefer B (BO)", bg="#f8d7da",
                  command=lambda: self._choice_var.set(2), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="TIE (Refine)", bg="#fff3cd",
                  command=lambda: self._choice_var.set(3), **btn_style).pack(side=tk.LEFT, padx=30)
        tk.Button(btn_frame, text="REJECT Both", bg="#e2e3e5",
                  command=lambda: self._choice_var.set(4), **btn_style).pack(side=tk.LEFT, padx=30)

        # This keeps Tk responsive and blocks until a button sets the IntVar
        self.root.update_idletasks()
        self.root.wait_variable(self._choice_var)

        return int(self._choice_var.get())
