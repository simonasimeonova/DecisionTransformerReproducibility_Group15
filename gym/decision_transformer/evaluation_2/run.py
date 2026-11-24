import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
from glob import glob

import gym
import numpy as np
import torch
import yaml

from .checkpoints import load_model, load_normalization
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg

CONFIG_DIR = Path(__file__).parent / "configs"


def load_settings(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with (CONFIG_DIR / "defaults.yaml").open() as file:
        defaults = yaml.safe_load(file)
    with config_path.open() as file:
        settings = yaml.safe_load(file)
    for section in ("evaluation", "model"):
        settings[section] = {**defaults[section], **settings.get(section, {})}

    # Paths in the YAML are relative to that YAML file.
    settings["dataset_path"] = str((config_path.parent / settings["dataset_path"]).resolve())
    for checkpoint in settings["checkpoints"].values():
        if checkpoint["path"]:
            path = str((config_path.parent / Path(checkpoint["path"]).expanduser()).resolve())
            matches = glob(path)
            if len(matches) > 1:
                raise ValueError(f"Multiple checkpoints match {path}; choose one run explicitly.")
            checkpoint["path"] = matches[0] if matches else path
    return settings


def completed_episodes(folder):
    completed = set()
    for path in folder.glob("*/episodes.csv"):
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                try:
                    completed.add((row["claim"], row["environment"], row["dataset"],
                                   row["checkpoint"], float(row["target_return"]), int(row["episode_seed"])))
                except (KeyError, TypeError, ValueError):
                    continue
    return completed


def run(settings, claim, output, completed):
    claims = ["return_conditioning", "context_comparison"]
    selected = claims if claim == "both" else [claims[int(claim) - 1]]
    selected = [name for name in selected if settings[name].get("enabled", True)]
    if not selected or any(not settings[name]["targets"] for name in selected):
        raise ValueError("Enable the selected claim and fill in its targets first.")

    evaluation, model_config = settings["evaluation"], settings["model"]
    if not evaluation["episode_seeds"]:
        raise ValueError("Fill in evaluation.episode_seeds first.")
    if not Path(settings["dataset_path"]).is_file():
        print(f"Skipping {settings['experiment_id']}: missing dataset {settings['dataset_path']}")
        return False
    pending = []
    for name in selected:
        for checkpoint_name in settings[name]["checkpoints"]:
            checkpoint = settings["checkpoints"][checkpoint_name]
            if not checkpoint["path"] or not Path(checkpoint["path"]).is_file():
                print(f"Skipping missing checkpoint: {checkpoint_name}")
                continue
            for target in settings[name]["targets"]:
                for seed in evaluation["episode_seeds"]:
                    key = (str(claims.index(name) + 1), settings["environment"], settings["dataset"],
                           checkpoint_name, float(target), int(seed))
                    if key not in completed:
                        pending.append((name, checkpoint_name, target, seed, key))
    if not pending:
        print(f"Nothing to evaluate for {settings['experiment_id']}")
        return False

    state_mean, state_std = load_normalization(settings["dataset_path"])
    try:
        env = gym.make(settings["env_id"])
    except gym.error.UnregisteredEnv:
        print(f"Skipping {settings['experiment_id']}: environment {settings['env_id']} is unavailable")
        return False
    output.mkdir(parents=True, exist_ok=False)
    with (output / "settings.yaml").open("w") as file:
        yaml.safe_dump(settings, file, sort_keys=False)

    try:
        with (output / "episodes.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["claim", "environment", "dataset", "checkpoint", "K",
                             "training_seed", "target_return", "episode_seed", "episode_return", "episode_length"])
            loaded_checkpoint = None
            for name, checkpoint_name, target, seed, key in pending:
                checkpoint = settings["checkpoints"][checkpoint_name]
                if checkpoint_name != loaded_checkpoint:
                    loaded_checkpoint = checkpoint_name
                    model = load_model(checkpoint, model_config, env, evaluation["device"])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.action_space.seed(seed)
                with torch.no_grad():
                    reward, length = evaluate_episode_rtg(
                        env, env.observation_space.shape[0], env.action_space.shape[0], model,
                        max_ep_len=evaluation["max_episode_steps"], device=evaluation["device"],
                        state_mean=state_mean, state_std=state_std,
                        scale=model_config["scale"], target_return=target / model_config["scale"],
                        mode=model_config["mode"],
                    )
                writer.writerow([claims.index(name) + 1, settings["environment"], settings["dataset"],
                                 checkpoint_name, checkpoint["K"], checkpoint["training_seed"],
                                 target, seed, reward, length])
                file.flush()
                completed.add(key)
                print(f"{name}: {checkpoint_name}, target={target}, seed={seed}, return={reward:.2f}", flush=True)
    finally:
        env.close()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--all", action="store_true", help="Run all six environment/dataset configurations")
    parser.add_argument("--claim", choices=["1", "2", "both"], required=True)
    parser.add_argument("--output", default="evaluation_results", help="Folder containing the results")
    args = parser.parse_args()

    if args.all:
        if args.config:
            parser.error("Use either --all or a config file.")
        configs = sorted(CONFIG_DIR.glob("*_medium_*.yaml"))
    elif args.config:
        configs = [Path(args.config)]
    else:
        parser.error("Provide a config file or --all.")

    results_folder = Path(args.output).expanduser()
    completed = completed_episodes(results_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for config in configs:
        settings = load_settings(config)
        output = results_folder / f"{settings['experiment_id']}_{timestamp}"
        if run(settings, args.claim, output, completed):
            print(f"Results saved to {output}")
