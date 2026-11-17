import argparse
from .load_settings import load_settings
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_file", help="Experiment YAML to preview")
    args = parser.parse_args()
    print(yaml.safe_dump(load_settings(args.experiment_file), sort_keys=False), end="")
