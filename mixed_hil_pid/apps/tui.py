#!/usr/bin/env python3
"""
Professional TUI for Mixed HIL-PID Optimization

A beautiful terminal interface for running and configuring PID optimization experiments.
Choose between different approaches and customize configurations easily.

Usage: python -m mixed_hil_pid.apps.tui
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import (
    Container,
    Horizontal,
    ScrollableContainer,
    Vertical,
)
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mixed_hil_pid.config_loader import get_pid_bounds, load_config


class ConfigEditorScreen(Screen):
    """Screen for editing configuration values."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("ctrl+s", "save_config", "Save"),
    ]

    def __init__(self, current_config: Dict[str, Any]):
        super().__init__()
        self.config = current_config.copy()
        self.inputs = {}

    def compose(self) -> ComposeResult:
        yield Header()

        with ScrollableContainer():
            yield Static("⚙️  Configuration Editor", classes="section-title")
            yield Static(
                "Edit PID optimization parameters below. Press Ctrl+S to save, Esc to cancel.",
                classes="help-text",
            )

            # PID Bounds
            yield Static("\n📐 PID Bounds", classes="subsection-title")
            with Vertical(classes="config-group"):
                for i, (name, bounds) in enumerate(
                    [
                        (
                            "Kp",
                            self.config.get(
                                "pid_bounds",
                                [[0.1, 10], [0.01, 10], [0.01, 10]],
                            )[0],
                        ),
                        (
                            "Ki",
                            self.config.get(
                                "pid_bounds",
                                [[0.1, 10], [0.01, 10], [0.01, 10]],
                            )[1],
                        ),
                        (
                            "Kd",
                            self.config.get(
                                "pid_bounds",
                                [[0.1, 10], [0.01, 10], [0.01, 10]],
                            )[2],
                        ),
                    ]
                ):
                    with Horizontal(classes="input-row"):
                        yield Label(f"{name} Range:", classes="input-label")
                        inp_min = Input(
                            value=str(bounds[0]),
                            placeholder=f"Min {name}",
                            classes="config-input",
                        )
                        inp_max = Input(
                            value=str(bounds[1]),
                            placeholder=f"Max {name}",
                            classes="config-input",
                        )
                        self.inputs[f"pid_bounds_{i}_min"] = inp_min
                        self.inputs[f"pid_bounds_{i}_max"] = inp_max
                        yield inp_min
                        yield inp_max

            # Simulation Settings
            yield Static(
                "\n🎮 Simulation Settings", classes="subsection-title"
            )
            with Vertical(classes="config-group"):
                for key, label in [
                    ("simulation_steps", "Simulation Steps"),
                    ("target_yaw_deg", "Target Yaw (deg)"),
                    ("dt", "Time Step (dt)"),
                ]:
                    with Horizontal(classes="input-row"):
                        yield Label(f"{label}:", classes="input-label")
                        inp = Input(
                            value=str(self.config.get(key, 0)),
                            placeholder=label,
                            classes="config-input",
                        )
                        self.inputs[key] = inp
                        yield inp

            # Performance Targets
            yield Static(
                "\n🎯 Performance Targets", classes="subsection-title"
            )
            with Vertical(classes="config-group"):
                for key, label in [
                    ("pid_max_overshoot_pct", "Max Overshoot (%)"),
                    ("pid_max_rise_time", "Max Rise Time (s)"),
                    ("pid_max_settling_time", "Max Settling Time (s)"),
                ]:
                    with Horizontal(classes="input-row"):
                        yield Label(f"{label}:", classes="input-label")
                        inp = Input(
                            value=str(self.config.get(key, 0)),
                            placeholder=label,
                            classes="config-input",
                        )
                        self.inputs[key] = inp
                        yield inp

            # Optimization Settings
            yield Static(
                "\n🔧 Optimization Settings", classes="subsection-title"
            )
            with Vertical(classes="config-group"):
                for key, label in [
                    ("max_iterations", "Max Iterations"),
                    ("base_mutation", "DE Mutation Factor"),
                    ("bo_pof_min", "BO PoF Minimum"),
                ]:
                    with Horizontal(classes="input-row"):
                        yield Label(f"{label}:", classes="input-label")
                        inp = Input(
                            value=str(
                                self.config.get(
                                    key, 1 if key == "num_experiments" else 0
                                )
                            ),
                            placeholder=label,
                            classes="config-input",
                        )
                        self.inputs[key] = inp
                        yield inp

            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button(
                    "💾 Save Configuration", variant="success", id="save-btn"
                )
                yield Button("❌ Cancel", variant="error", id="cancel-btn")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self.action_save_config()
        elif event.button.id == "cancel-btn":
            self.app.pop_screen()

    def action_save_config(self) -> None:
        """Save configuration changes."""
        try:
            # Update config with new values
            pid_bounds = []
            for i in range(3):
                min_val = float(self.inputs[f"pid_bounds_{i}_min"].value)
                max_val = float(self.inputs[f"pid_bounds_{i}_max"].value)
                pid_bounds.append([min_val, max_val])

            # Update app's config
            self.app.current_config["pid_bounds"] = pid_bounds

            for key in [
                "simulation_steps",
                "target_yaw_deg",
                "dt",
                "pid_max_overshoot_pct",
                "pid_max_rise_time",
                "pid_max_settling_time",
                "max_iterations",
                "base_mutation",
                "bo_pof_min",
            ]:
                if key in self.inputs:
                    self.app.current_config[key] = float(
                        self.inputs[key].value
                    )

            self.app.notify(
                "✅ Configuration updated! Changes will apply on next run.",
                severity="information",
            )
            self.app.pop_screen()

        except Exception as e:
            self.app.notify(
                f"❌ Error saving config: {str(e)}", severity="error"
            )


class ExperimentCountScreen(ModalScreen):
    """Screen for inputting number of experiments and robot selection."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Cancel"),
    ]

    def __init__(self, approach_name: str):
        super().__init__()
        self.approach_name = approach_name
        self.count = 1
        self.robot_type = "husky"

    def compose(self) -> ComposeResult:
        yield Header()

        with ScrollableContainer():
            yield Static(
                f"🚀 Launch {self.approach_name}", classes="section-title"
            )
            yield Static(
                "Configure experiment parameters", classes="help-text"
            )

            with Vertical(classes="config-group"):
                # Robot selection
                with Horizontal(classes="input-row"):
                    yield Label("Robot Type:", classes="input-label")
                    yield Select(
                        [
                            ("Husky (Differential Drive)", "husky"),
                            ("Ackermann (Racecar)", "ackermann"),
                        ],
                        value="husky",
                        classes="config-input",
                        id="robot-select",
                    )

                # Experiment count
                with Horizontal(classes="input-row"):
                    yield Label(
                        "Number of Experiments:", classes="input-label"
                    )
                    yield Input(
                        value="1",
                        placeholder="Enter number",
                        classes="config-input",
                        id="exp-count-input",
                    )

            with Horizontal(classes="button-row"):
                yield Button("▶️  Run", variant="success", id="run-btn")
                yield Button("❌ Cancel", variant="error", id="cancel-btn")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            try:
                count_input = self.query_one("#exp-count-input", Input)
                robot_select = self.query_one("#robot-select", Select)
                self.count = max(1, int(count_input.value))
                self.robot_type = robot_select.value
                self.dismiss((self.count, self.robot_type))
            except ValueError:
                self.app.notify(
                    "❌ Please enter a valid number", severity="error"
                )
        elif event.button.id == "cancel-btn":
            self.dismiss(None)


class HILApp(App):
    """Professional TUI for Mixed HIL-PID Optimization."""

    CSS = """
    Screen {
        background: $surface;
    }
    
    .section-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1;
        padding: 1;
    }
    
    .subsection-title {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
    }
    
    .help-text {
        text-align: center;
        color: $text-muted;
        margin-bottom: 2;
    }
    
    .approach-card {
        border: tall $primary;
        background: $surface-darken-1;
        padding: 1 2;
        margin: 1;
        height: auto;
    }
    
    .approach-title {
        text-style: bold;
        color: $accent;
    }
    
    .approach-desc {
        color: $text-muted;
        margin-top: 1;
    }
    
    .config-preview {
        border: round $secondary;
        background: $surface;
        padding: 1;
        margin: 1;
        height: auto;
    }
    
    .config-group {
        padding: 0 2;
    }
    
    .input-row {
        height: auto;
        margin: 1 0;
    }
    
    .input-label {
        width: 30;
        content-align: right middle;
        color: $text;
    }
    
    .config-input {
        width: 20;
        margin-left: 2;
    }
    
    .button-row {
        align: center middle;
        height: auto;
        margin: 2;
    }
    
    .button-row Button {
        margin: 0 2;
    }
    
    DataTable {
        margin: 1;
        height: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("e", "edit_config", "Edit Config"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    TITLE = "Mixed HIL-PID Optimization"
    SUB_TITLE = "Professional PID Controller Tuning"

    def __init__(self):
        super().__init__()
        self.current_config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load current configuration."""
        config = load_config()
        pid_bounds = get_pid_bounds(config)

        # Get robot-specific values (default to husky)
        robot_type = config.get("robot_type", "husky")
        robot_config = config["robots"][robot_type]

        return {
            "pid_bounds": pid_bounds,
            "simulation_steps": config["simulation_steps"],
            "dt": config["dt"],
            "target_yaw_deg": config["target_yaw_deg"],
            "pid_max_overshoot_pct": robot_config["pid_max_overshoot_pct"],
            "pid_max_rise_time": robot_config["pid_max_rise_time"],
            "pid_max_settling_time": robot_config["pid_max_settling_time"],
            "max_iterations": config["max_iterations"],
            "base_mutation": config["base_mutation"],
            "bo_pof_min": config["bo_pof_min"],
        }

    def compose(self) -> ComposeResult:
        yield Header()

        with ScrollableContainer():
            # Welcome
            yield Static(
                "🚀 Mixed HIL-PID Optimization Suite", classes="section-title"
            )
            yield Static(
                "Choose your optimization approach and configure parameters",
                classes="help-text",
            )

            # Approach Selection
            with TabbedContent(initial="approaches"):
                # Tab 1: Approaches
                with TabPane("🎯 Approaches", id="approaches"):
                    yield from self.render_approaches()

                # Tab 2: Configuration
                with TabPane("⚙️  Configuration", id="configuration"):
                    yield from self.render_config_preview()

        yield Footer()

    def render_approaches(self) -> ComposeResult:
        """Render approach selection cards."""

        approaches = [
            {
                "id": "mixed_hil",
                "title": "🔄 Mixed HIL (DE vs BO)",
                "desc": "Compare DE and BO candidates side-by-side\n4 feedback options: Prefer DE, Prefer BO, TIE, REJECT",
                "cmd": "mixed_hil_rerun.py",
                "variant": "primary",
            },
            {
                "id": "de_hil",
                "title": "🔷 DE HIL (Differential Evolution)",
                "desc": "Human-guided Differential Evolution\\n2 feedback options: ACCEPT (refine), REJECT (expand)",
                "cmd": "de_hil_rerun.py",
                "variant": "success",
            },
            {
                "id": "bo_hil",
                "title": "🔶 BO HIL (Bayesian Optimization)",
                "desc": "Human-guided Bayesian Optimization\\n2 feedback options: ACCEPT (refine), REJECT (expand)",
                "cmd": "bo_hil_rerun.py",
                "variant": "warning",
            },
            {
                "id": "de_vanilla",
                "title": "⚡ DE Vanilla (No GUI)",
                "desc": "Automated DE optimization for benchmarking\\nRuns experiments automatically",
                "cmd": "de_vanilla_rerun.py",
                "variant": "default",
            },
            {
                "id": "bo_vanilla",
                "title": "⚡ BO Vanilla (No GUI)",
                "desc": "Automated BO optimization for benchmarking\\nRuns experiments automatically",
                "cmd": "bo_vanilla_rerun.py",
                "variant": "default",
            },
        ]

        for approach in approaches:
            with Container(classes="approach-card"):
                yield Static(approach["title"], classes="approach-title")
                yield Static(approach["desc"], classes="approach-desc")
                yield Button(
                    f"▶️  Run {approach['id'].upper()}",
                    variant=approach["variant"],
                    id=f"run-{approach['id']}",
                )

    def render_config_preview(self) -> ComposeResult:
        """Render configuration preview."""
        with Container(classes="config-preview"):
            yield Static("Current Configuration", classes="approach-title")

            table = DataTable()
            table.add_columns("Parameter", "Value")

            table.add_row(
                "PID Bounds (Kp)", f"{self.current_config['pid_bounds'][0]}"
            )
            table.add_row(
                "PID Bounds (Ki)", f"{self.current_config['pid_bounds'][1]}"
            )
            table.add_row(
                "PID Bounds (Kd)", f"{self.current_config['pid_bounds'][2]}"
            )
            table.add_row(
                "Simulation Steps",
                f"{self.current_config['simulation_steps']}",
            )
            table.add_row(
                "Target Yaw", f"{self.current_config['target_yaw_deg']}°"
            )
            table.add_row(
                "Max Overshoot",
                f"{self.current_config['pid_max_overshoot_pct']}%",
            )
            table.add_row(
                "Max Rise Time", f"{self.current_config['pid_max_rise_time']}s"
            )
            table.add_row(
                "Max Settling Time",
                f"{self.current_config['pid_max_settling_time']}s",
            )
            table.add_row(
                "Max Iterations", f"{self.current_config['max_iterations']}"
            )
            table.add_row(
                "DE Mutation Factor", f"{self.current_config['base_mutation']}"
            )
            table.add_row("BO PoF Min", f"{self.current_config['bo_pof_min']}")

            yield table
            yield Button(
                "✏️  Edit Configuration",
                variant="primary",
                id="edit-config-btn",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id and button_id.startswith("run-"):
            approach = button_id.replace("run-", "")
            self.prompt_and_run(approach)
        elif button_id == "edit-config-btn":
            self.action_edit_config()

    def prompt_and_run(self, approach: str) -> None:
        """Prompt for experiment count and robot selection, then run approach."""
        approach_names = {
            "mixed_hil": "Mixed HIL",
            "de_hil": "DE HIL",
            "bo_hil": "BO HIL",
            "de_vanilla": "DE Vanilla",
            "bo_vanilla": "BO Vanilla",
        }

        def handle_result(result):
            if result is not None:
                count, robot_type = result
                self.run_approach(approach, count, robot_type)

        self.push_screen(
            ExperimentCountScreen(approach_names.get(approach, approach)),
            handle_result,
        )

    def run_approach(
        self,
        approach: str,
        num_experiments: int = 1,
        robot_type: str = "husky",
    ) -> None:
        """Run the selected optimization approach with current config."""
        # Get paths
        project_root = Path(__file__).parent.parent.parent
        scripts_dir = project_root / "mixed_hil_pid" / "scripts"
        config_file = project_root / "config.yaml"
        venv_python = "/home/waleed/python-env/bin/python"

        approach_files = {
            "mixed_hil": "mixed_hil_rerun.py",
            "de_hil": "de_hil_rerun.py",
            "bo_hil": "bo_hil_rerun.py",
            "de_vanilla": "de_vanilla_rerun.py",
            "bo_vanilla": "bo_vanilla_rerun.py",
        }

        if approach not in approach_files:
            self.notify(f"❌ Unknown approach: {approach}", severity="error")
            return

        script_path = scripts_dir / approach_files[approach]

        if not script_path.exists():
            self.notify(
                f"❌ Script not found: {script_path}", severity="error"
            )
            return

        # Save current config to config.yaml
        try:
            self.save_config_to_file(config_file)
            self.notify(f"✅ Configuration saved", severity="information")
        except Exception as e:
            self.notify(
                f"⚠️ Could not save config: {str(e)}", severity="warning"
            )

        self.notify(
            f"🚀 Launching {approach.upper()} with {robot_type} robot... (TUI will minimize)",
            severity="information",
        )

        try:
            # Launch script with experiment count and robot arguments
            process = subprocess.Popen(
                [
                    venv_python,
                    str(script_path),
                    str(num_experiments),
                    "--robot",
                    robot_type,
                ],
                cwd=str(project_root),
                # Don't redirect - let GUI show!
            )

            # Exit TUI to let the GUI take over
            self.exit(
                message=f"✅ {approach.upper()} running {num_experiments} experiment(s) on {robot_type} (PID: {process.pid})"
            )

        except Exception as e:
            self.notify(
                f"❌ Error launching {approach}: {str(e)}", severity="error"
            )

    def save_config_to_file(self, config_file: Path) -> None:
        """Save current config to config.yaml file."""
        # Load existing config to preserve robot configurations
        try:
            existing_config = load_config()
        except:
            existing_config = {}

        # Convert config to YAML format, preserving robot configurations
        yaml_config = {
            "robot_type": existing_config.get("robot_type", "husky"),
            "robots": existing_config.get("robots", {}),
            "pid_bounds": {
                "kp": list(self.current_config["pid_bounds"][0]),
                "ki": list(self.current_config["pid_bounds"][1]),
                "kd": list(self.current_config["pid_bounds"][2]),
            },
            "simulation_steps": int(self.current_config["simulation_steps"]),
            "dt": float(self.current_config["dt"]),
            "base_mutation": float(self.current_config["base_mutation"]),
            "preference_lr": 0.3,  # Not exposed in TUI, keep default
            "max_iterations": int(self.current_config["max_iterations"]),
            "target_yaw_deg": float(self.current_config["target_yaw_deg"]),
            "pid_max_overshoot_pct": int(
                self.current_config["pid_max_overshoot_pct"]
            ),
            "pid_max_rise_time": int(self.current_config["pid_max_rise_time"]),
            "pid_max_settling_time": int(
                self.current_config["pid_max_settling_time"]
            ),
            "display_realtime": False,  # Not exposed in TUI, keep default
            "pid_sat_penalty": 0.01,  # Not exposed in TUI, keep default
            "pid_strict_output_limit": True,  # Not exposed in TUI, keep default
            "pid_sat_hard_penalty": 10000.0,  # Not exposed in TUI, keep default
            "bo_pof_min": float(self.current_config["bo_pof_min"]),
        }

        # Write YAML file
        with config_file.open("w") as f:
            yaml.dump(
                yaml_config, f, default_flow_style=False, sort_keys=False
            )

    def action_edit_config(self) -> None:
        """Open configuration editor."""
        self.push_screen(ConfigEditorScreen(self.current_config))

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def main():
    """Entry point for the TUI application."""
    app = HILApp()
    app.run()


if __name__ == "__main__":
    main()
