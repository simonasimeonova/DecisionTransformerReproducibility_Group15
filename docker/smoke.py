"""Build-time checks: CPU simulation and the actual training model imports."""
import sys

import gym
import torch
import wandb

sys.path.insert(0, "/opt/decision-transformer/gym")
from decision_transformer.models.decision_transformer import DecisionTransformer

env = gym.make("Ant-v3")
try:
    env.seed(3)
    obs = env.reset()
    obs, reward, done, info = env.step(env.action_space.sample())
    model = DecisionTransformer(
        state_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        hidden_size=128,
        max_length=1,
        max_ep_len=1000,
        n_layer=3,
        n_head=1,
    )
    print("Ant OK:", obs.shape, "torch:", torch.__version__, "CUDA build:", torch.version.cuda)
finally:
    env.close()
