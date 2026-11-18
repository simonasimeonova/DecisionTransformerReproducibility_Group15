"""Evaluate trained models and save one CSV row per episode."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import random

import gym
import numpy as np
import torch
import yaml

from .checkpoints import load_model, load_normalization
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg


def load_settings(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with (Path(__file__).parent / "configs/defaults.yaml").open() as file:
        defaults = yaml.safe_load(file)
    with config_path.open() as file:
        settings = yaml.safe_load(file)
    for section in ("evaluation", "model"):
        settings[section] = {**defaults[section], **settings.get(section, {})}

    # Paths in the YAML are relative to that YAML file.
    settings["dataset_path"] = str((config_path.parent / settings["dataset_path"]).resolve())
    for checkpoint in settings["checkpoints"].values():
        if checkpoint["path"]:
            checkpoint["path"] = str((config_path.parent / Path(checkpoint["path"]).expanduser()).resolve())
    return settings


def run(settings, claim, output):
    claims = ["return_conditioning", "context_comparison"]
    selected = claims if claim == "both" else [claims[int(claim) - 1]]
    selected = [name for name in selected if settings[name].get("enabled", True)]
    if not selected or any(not settings[name]["targets"] for name in selected):
        raise ValueError("Enable the selected claim and fill in its targets first.")

    evaluation, model_config = settings["evaluation"], settings["model"]
    state_mean, state_std = load_normalization(settings["dataset_path"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "settings.yaml").open("w") as file:
        yaml.safe_dump(settings, file, sort_keys=False)

    env = gym.make(settings["env_id"])
    try:
        with (output / "episodes.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["claim", "environment", "dataset", "checkpoint", "K",
                             "training_seed", "target_return", "episode_seed", "episode_return", "episode_length"])
            for name in selected:
                for checkpoint_name in settings[name]["checkpoints"]:
                    checkpoint = settings["checkpoints"][checkpoint_name]
                    model = load_model(checkpoint, model_config, env, evaluation["device"])
                    for target in settings[name]["targets"]:
                        for seed in evaluation["episode_seeds"]:
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
                            print(f"{name}: {checkpoint_name}, target={target}, seed={seed}, return={reward:.2f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--claim", choices=["1", "2", "both"], required=True)
    parser.add_argument("--output", help="New directory for this run's results")
    args = parser.parse_args()
    settings = load_settings(args.config)
    output = Path(args.output).expanduser() if args.output else Path("evaluation_results") / (
        settings["experiment_id"] + datetime.now().strftime("_%Y%m%d_%H%M%S_%f"))
    run(settings, args.claim, output)
    print(f"Results saved to {output}")
