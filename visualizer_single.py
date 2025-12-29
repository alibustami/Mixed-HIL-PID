import tkinter as tk
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class SingleFeedbackGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HIL Validation Dashboard")
        self.root.geometry("1100x750")
        self._choice_var = tk.IntVar(value=0)  # 0=none, 1=Accept, 2=Reject

    def show_candidate(self, history, params, fit, metrics, prev_histories, label_text="Candidate"):
        self._choice_var.set(0)

        for widget in self.root.winfo_children():
            widget.destroy()

        # --- Header Info ---
        header_frame = tk.Frame(self.root)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        def format_metrics(m):
            settle = f"{m['settling_time']:.2f}s" if m["settling_time"] > 0 else "Unstable"
            rise = f"{m['rise_time']:.2f}s" if m["rise_time"] > 0 else "N/A"
            return f"Overshoot: {m['overshoot']:.1f}% | Rise Time: {rise} | Settling Time: {settle}"

        info_text = f"{label_text}\nKp={params[0]:.2f}, Ki={params[1]:.2f}, Kd={params[2]:.2f}\nFitness: {fit:.4f}\n{format_metrics(metrics)}"
        
        lbl = tk.Label(
            header_frame,
            text=info_text,
            font=("Arial", 12, "bold"), bg="#e6f2ff", relief=tk.RAISED, padx=10, pady=10
        )
        lbl.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5)

        # --- Plotting ---
        sns.set_theme(style="whitegrid")
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        # Plot Target
        ax.axhline(y=90, color="k", linestyle="--", alpha=0.5, label="Target")

        # Plot Previous (Faded)
        n_prev = len(prev_histories)
        if n_prev > 0:
            colors = sns.color_palette("Greys", n_colors=max(n_prev + 2, 3))
            for idx, hist in enumerate(prev_histories):
                # Fade older iterations
                alpha = 0.2 + 0.4 * (idx / n_prev)
                ax.plot(hist["time"], hist["actual"], color=colors[idx], linewidth=1.0, alpha=alpha, zorder=1)

        # Plot Current
        ax.plot(history["time"], history["actual"], color="blue", linewidth=2.5, label="Current", zorder=2)

        ax.set_title(f"Response: {label_text}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Yaw (Deg)")
        ax.set_ylim(-10, 120)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Buttons ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        btn_style = {"font": ("Arial", 12, "bold"), "width": 20, "height": 2}

        # ACCEPT -> Triggers Refine logic
        tk.Button(btn_frame, text="ACCEPT (Refine)", bg="#d1e7dd", fg="#0f5132",
                  command=lambda: self._choice_var.set(1), **btn_style).pack(side=tk.LEFT, padx=50, expand=True)

        # REJECT -> Triggers Expand logic
        tk.Button(btn_frame, text="REJECT (Expand)", bg="#f8d7da", fg="#842029",
                  command=lambda: self._choice_var.set(2), **btn_style).pack(side=tk.RIGHT, padx=50, expand=True)

        self.root.update_idletasks()
        self.root.wait_variable(self._choice_var)

        return int(self._choice_var.get())