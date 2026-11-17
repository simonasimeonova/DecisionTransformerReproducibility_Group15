"""Read an experiment YAML and apply shared evaluation defaults."""

import argparse
from pathlib import Path

import yaml


def load_settings(experiment_file):
    experiment_file = Path(experiment_file).expanduser().resolve()
    defaults_file = Path(__file__).parent / "configs" / "defaults.yaml"

    with defaults_file.open() as file:
        defaults = yaml.safe_load(file)
    with experiment_file.open() as file:
        settings = yaml.safe_load(file)

    # Experiment settings override shared defaults.
    settings["evaluation"] = {
        **defaults["evaluation"],
        **settings.get("evaluation", {}),
    }

    # Interpret checkpoint paths relative to the experiment YAML.
    for name, path in settings.get("checkpoints", {}).items():
        if path:
            path = Path(path).expanduser()
            settings["checkpoints"][name] = str((experiment_file.parent / path).resolve())

    return settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_file", help="Path to the experiment YAML")
    args = parser.parse_args()
    print(yaml.safe_dump(load_settings(args.experiment_file), sort_keys=False), end="")
