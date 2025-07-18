"""
training configuration TUI.
fully ai generated
"""
import subprocess
from datetime import datetime
from pathlib import Path

import tomli_w
from fastcore.script import call_parse
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import Button, Footer, Header, Input, Label, Select, Switch


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
        self.defaults = {
            "backbone": "dinov2_vitb14_reg",
            "input_type": "all",
            "wandb_name": "metarep",
            "embedding": False,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "num_layers": 6,
            "hidden_act": "gelu",
            "bias": False,
            "logit_bias": True,
            "attention_dropout": 0.1,
            "sequence_length": 100,
            "batch_size": 64,
            "training_steps": 1000000,
            "seed": 1234,
            "lr": 3e-5,
            "weight_decay": 1e-4,
            "warmup_steps": 50000,
            "num_components": None,
            "constant_lr": False,
            "log_interval_steps": 10,
            "eval_interval_steps": 100,
            "num_eval_episodes": 128,
            "eval_dims": [0, 1, 2],
            "tags": [],
            "checkpoint_dir": "checkpoints",
            "checkpoint_interval_steps": 1000,
            "scale": True,
            "spose_input": False,
            "fixed_label": False,
            "weighted": False,
            "positional_embedding_type": "learned",
            "compile": False,
        }

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(classes="container"):
            yield Label("Training Configuration", id="title")

            with ScrollableContainer():
                yield self._create_input_group("name", "Name", "Name for the run")
                yield self._create_input_group("backbone", "Backbone", self.defaults["backbone"])
                yield self._create_select_group("input_type", "Input Type", ["all", "cls", "register", "patch"], self.defaults["input_type"])
                yield self._create_input_group("wandb_name", "Wandb Project", self.defaults["wandb_name"])

                yield self._create_switch_group("embedding", "Use Embedding", self.defaults["embedding"])
                yield self._create_input_group("hidden_size", "Hidden Size", str(self.defaults["hidden_size"]))
                yield self._create_input_group("num_attention_heads", "Attention Heads", str(self.defaults["num_attention_heads"]))
                yield self._create_input_group("intermediate_size", "Intermediate Size", str(self.defaults["intermediate_size"]))
                yield self._create_input_group("num_layers", "Num Layers", str(self.defaults["num_layers"]))
                yield self._create_select_group("hidden_act", "Hidden Activation", ["gelu", "relu", "swish"], self.defaults["hidden_act"])

                yield self._create_switch_group("bias", "Use Bias", self.defaults["bias"])
                yield self._create_switch_group("logit_bias", "Logit Bias", self.defaults["logit_bias"])
                yield self._create_input_group("attention_dropout", "Attention Dropout", str(self.defaults["attention_dropout"]))
                yield self._create_input_group("sequence_length", "Sequence Length", str(self.defaults["sequence_length"]))

                yield self._create_input_group("batch_size", "Batch Size", str(self.defaults["batch_size"]))
                yield self._create_input_group("training_steps", "Training Steps", str(self.defaults["training_steps"]))
                yield self._create_input_group("seed", "Seed", str(self.defaults["seed"]))
                yield self._create_input_group("lr", "Learning Rate", str(self.defaults["lr"]))
                yield self._create_input_group("weight_decay", "Weight Decay", str(self.defaults["weight_decay"]))
                yield self._create_input_group("warmup_steps", "Warmup Steps", str(self.defaults["warmup_steps"]))

                yield self._create_input_group("num_components", "Num Components (PCA)", "")
                yield self._create_switch_group("constant_lr", "Constant LR", self.defaults["constant_lr"])
                yield self._create_input_group("log_interval_steps", "Log Interval Steps", str(self.defaults["log_interval_steps"]))
                yield self._create_input_group("eval_interval_steps", "Eval Interval Steps", str(self.defaults["eval_interval_steps"]))
                yield self._create_input_group("num_eval_episodes", "Num Eval Episodes", str(self.defaults["num_eval_episodes"]))
                yield self._create_input_group("eval_dims", "Eval Dims (space separated)", " ".join(map(str, self.defaults["eval_dims"])))
                yield self._create_input_group("tags", "Tags (space separated)", " ".join(self.defaults["tags"]))

                yield self._create_input_group("checkpoint_dir", "Checkpoint Dir", self.defaults["checkpoint_dir"])
                yield self._create_input_group("checkpoint_interval_steps", "Checkpoint Interval", str(self.defaults["checkpoint_interval_steps"]))

                yield self._create_switch_group("scale", "Scale Input", self.defaults["scale"])
                yield self._create_switch_group("spose_input", "SPoSE Input", self.defaults["spose_input"])
                yield self._create_switch_group("fixed_label", "Fixed Label", self.defaults["fixed_label"])
                yield self._create_switch_group("weighted", "Weighted Sampling", self.defaults["weighted"])
                yield self._create_select_group("positional_embedding_type", "Positional Embedding", ["learned", "rope", "sinusoidal"], self.defaults["positional_embedding_type"])
                yield self._create_switch_group("compile", "Compile Model", self.defaults["compile"])

                with Container(classes="submit-container"):
                    yield Button("Submit Training Job", id="submit", classes="submit-button")
                    yield Button("Save Config Only", id="save_config", classes="submit-button")

        yield Footer()

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

        for key in self.defaults.keys():
            widget = self.query_one(f"#{key}")

            if isinstance(widget, Input):
                value = widget.value.strip()
                if value:
                    if key in [
                        "hidden_size",
                        "num_attention_heads",
                        "intermediate_size",
                        "num_layers",
                        "sequence_length",
                        "batch_size",
                        "training_steps",
                        "seed",
                        "warmup_steps",
                        "log_interval_steps",
                        "eval_interval_steps",
                        "num_eval_episodes",
                        "checkpoint_interval_steps",
                    ]:
                        config[key] = int(value)
                    elif key in ["lr", "weight_decay", "attention_dropout"]:
                        config[key] = float(value)
                    elif key == "num_components":
                        config[key] = int(value) if value else None
                    elif key == "eval_dims":
                        config[key] = [int(x.strip()) for x in value.split()]
                    elif key == "tags":
                        config[key] = value.split()
                    else:
                        config[key] = value

            elif isinstance(widget, Switch):
                config[key] = widget.value

            elif isinstance(widget, Select):
                if widget.value != Select.BLANK:
                    config[key] = widget.value

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
