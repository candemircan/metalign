"""
training configuration TUI.
fully ai generated
"""
import inspect
import subprocess
from datetime import datetime
from pathlib import Path

import tomli_w
from fastcore.script import call_parse
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import Button, Footer, Header, Input, Label, Select, Switch
from train import main as train_main


class TrainingConfigTUI(App):
    CSS = """
    .container {
        padding: 1;
    }
    
    .input-group {
        height: 3;
        margin: 0 1;
    }
    
    .label {
        width: 25;
        text-align: right;
        padding-right: 1;
    }
    
    .input {
        width: 1fr;
    }
    
    .submit-container {
        height: 5;
        align: center middle;
        margin: 1;
    }
    
    .submit-button {
        width: 20;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+s", "submit", "Submit"),
    ]

    def __init__(self):
        super().__init__()
        self.config_values = {}
        sig = inspect.signature(train_main)
        self.parameters = {}
        self.defaults = {}
        
        for param_name, param in sig.parameters.items():
            self.parameters[param_name] = param
            if param.default != inspect.Parameter.empty:
                self.defaults[param_name] = param.default
            else:
                self.defaults[param_name] = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(classes="container"):
            yield Label("Training Configuration", id="title")

            with ScrollableContainer():
                for param_name, param in self.parameters.items():
                    default_val = self.defaults.get(param_name)
                    
                    if param_name == "name":
                        yield self._create_input_group(param_name, "Name", "Name for the run")
                    elif param_name == "input_type":
                        yield self._create_select_group(param_name, "Input Type", ["all", "cls", "register", "patch"], default_val)
                    elif param_name == "hidden_act":
                        yield self._create_select_group(param_name, "Hidden Activation", ["gelu", "relu", "swish"], default_val)
                    elif param_name == "positional_embedding_type":
                        yield self._create_select_group(param_name, "Positional Embedding", ["learned", "rope", "sinusoidal"], default_val)
                    elif param.annotation == "bool_arg" or (default_val is not None and isinstance(default_val, bool)):
                        yield self._create_switch_group(param_name, self._format_label(param_name), default_val if default_val is not None else False)
                    else:
                        placeholder = str(default_val) if default_val is not None else ""
                        if param_name in ["eval_dims", "tags"] and isinstance(default_val, list):
                            placeholder = " ".join(map(str, default_val))
                        yield self._create_input_group(param_name, self._format_label(param_name), placeholder)

                with Container(classes="submit-container"):
                    yield Button("Submit Training Job", id="submit", classes="submit-button")
                    yield Button("Save Config Only", id="save_config", classes="submit-button")

        yield Footer()
    
    def _format_label(self, param_name: str) -> str:
        """Convert parameter name to a formatted label"""
        return " ".join(word.capitalize() for word in param_name.split("_"))

    def _create_input_group(self, key: str, label: str, placeholder: str = "") -> Container:
        return Horizontal(Label(f"{label}:", classes="label"), Input(placeholder=placeholder, id=key, classes="input"), classes="input-group")

    def _create_switch_group(self, key: str, label: str, default: bool = False) -> Container:
        return Horizontal(Label(f"{label}:", classes="label"), Switch(value=default, id=key), classes="input-group")

    def _create_select_group(self, key: str, label: str, options: list, default: str = "") -> Container:
        select_options = [(option, option) for option in options]
        return Horizontal(Label(f"{label}:", classes="label"), Select(options=select_options, value=default, id=key, classes="input"), classes="input-group")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.action_submit()
        elif event.button.id == "save_config":
            self.action_save_config()

    def action_submit(self) -> None:
        config_file = self._generate_config()
        if config_file:
            try:
                subprocess.run(["sbatch", "bin/train.slurm", str(config_file)], check=True)
                self.notify(f"Training job submitted with config: {config_file}")
            except subprocess.CalledProcessError as e:
                self.notify(f"Failed to submit job: {e}", severity="error")
            except FileNotFoundError:
                self.notify("sbatch command not found. Make sure you're on a SLURM system.", severity="error")

    def action_save_config(self) -> None:
        config_file = self._generate_config()
        if config_file:
            self.notify(f"Configuration saved to: {config_file}")

    def _generate_config(self) -> Path | None:
        config = {}

        for param_name, param in self.parameters.items():
            if param_name == "name":
                continue
            
            widget = self.query_one(f"#{param_name}")
            default_val = self.defaults.get(param_name)

            if isinstance(widget, Input):
                value = widget.value.strip()
                if value:
                    if param.annotation is int or (default_val is not None and isinstance(default_val, int)):
                        config[param_name] = int(value) if value != "None" else None
                    elif param.annotation is float or (default_val is not None and isinstance(default_val, float)):
                        config[param_name] = float(value)
                    elif param_name == "eval_dims":
                        config[param_name] = [int(x.strip()) for x in value.split()]
                    elif param_name == "tags":
                        config[param_name] = value.split()
                    elif param_name == "num_components" and value.lower() in ["none", ""]:
                        config[param_name] = None
                    else:
                        config[param_name] = value

            elif isinstance(widget, Switch):
                config[param_name] = widget.value

            elif isinstance(widget, Select):
                if widget.value != Select.BLANK:
                    config[param_name] = widget.value

        name_widget = self.query_one("#name")
        run_name = name_widget.value.strip() if name_widget.value.strip() else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if "name" not in config and run_name != f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}":
            config["name"] = run_name

        config_dir = Path("data/configs")
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / f"{run_name}.toml"

        with open(config_file, "wb") as f:
            tomli_w.dump(config, f)

        return config_file


@call_parse
def main():
    """Launch TUI for configuring and submitting training jobs"""
    app = TrainingConfigTUI()
    app.run()
