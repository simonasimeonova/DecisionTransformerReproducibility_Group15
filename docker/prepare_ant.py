"""Prepare only Ant medium-replay v2, without installing all D4RL envs.

URL: D4RL d4rl/infos.py, Gym-MuJoCo V2 loop. Episode splitting matches
gym/data/download_d4rl_datasets.py: include terminal/timeout transitions;
discard the final unfinished trajectory. No reward/observation changes.
"""
import argparse
import os
import pickle
import urllib.request
from pathlib import Path

import h5py
import numpy as np

URL = "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/ant_medium_replay-v2.hdf5"
KEYS = ("observations", "next_observations", "actions", "rewards", "terminals")


def convert(source, target):
    with h5py.File(source, "r") as handle:
        data = {key: handle[key][:] for key in KEYS}
        timeouts = handle["timeouts"][:] if "timeouts" in handle else None
    paths = []
    start = 0
    episode_step = 0
    for i in range(len(data["rewards"])):
        timeout = bool(timeouts[i]) if timeouts is not None else episode_step == 999
        if bool(data["terminals"][i]) or timeout:
            paths.append({key: value[start:i + 1] for key, value in data.items()})
            start = i + 1
            episode_step = 0
        episode_step += 1
    if not paths:
        raise ValueError("Dataset contains no complete trajectories")
    temporary = target.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(paths, stream, protocol=4)
    temporary.replace(target)
    returns = np.array([path["rewards"].sum() for path in paths])
    print("Saved", target, "trajectories:", len(paths), "return mean/max:", returns.mean(), returns.max(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / "ant-medium-replay-v2.pkl"
    if target.exists():
        print("Using prepared dataset:", target, flush=True)
        return
    source = Path(os.environ.get("ANT_HDF5_PATH", str(args.directory / "ant_medium_replay-v2.hdf5")))
    if not source.exists():
        if "ANT_HDF5_PATH" in os.environ:
            raise FileNotFoundError(source)
        print("Downloading", URL, flush=True)
        temporary = source.with_suffix(".hdf5.tmp")
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        temporary.replace(source)
    convert(source, target)


if __name__ == "__main__":
    main()
