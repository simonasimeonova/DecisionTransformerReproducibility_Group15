"""Adapter for the raw state_dict checkpoints saved by experiment.py."""
import pickle
import numpy as np
import torch
from decision_transformer.models.decision_transformer import DecisionTransformer


def load_normalization(dataset_path):
    # Match experiment.py: all observations, including when pct_traj < 1.
    with open(dataset_path, "rb") as file:
        trajectories = pickle.load(file)
    states = np.concatenate([path["observations"] for path in trajectories], axis=0)
    return states.mean(axis=0), states.std(axis=0) + 1e-6


def load_model(checkpoint, config, env, device):
    model = DecisionTransformer(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        max_length=checkpoint["K"],
        max_ep_len=config["max_ep_len"],
        hidden_size=config["embed_dim"],
        n_layer=config["n_layer"], n_head=config["n_head"],
        n_inner=4 * config["embed_dim"],
        activation_function=config["activation_function"],
        n_positions=1024,
        resid_pdrop=config["dropout"], attn_pdrop=config["dropout"],
    )
    model.load_state_dict(torch.load(checkpoint["path"], map_location="cpu"))
    return model.to(device).eval()
