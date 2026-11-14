# How to Run Decision Transformer Experiments

## 1. Project structure

All experiment-related files are in the `gym/` directory.

| Path | Contents |
|---|---|
| `experiment.py` | Main training and evaluation script |
| `decision_transformer/` | Decision Transformer implementation |
| `data/` | Prepared D4RL datasets |
| `checkpoints/` | Saved trained model weights |
| `data/download_d4rl_datasets.py` | Dataset preparation script |

## 2. Environment

The experiments use the `decision-transformer` Micromamba environment.

Check that it exists:

```bash
~/bin/micromamba env list
```

Activate the required MuJoCo library paths in your terminal:

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:$HOME/.local/share/mamba/envs/decision-transformer/lib:$LD_LIBRARY_PATH"
```

Commands must clear the global Python environment variables to avoid conflicts:

```bash
env -u PYTHONPATH -u PYTHONHOME ~/bin/micromamba run -n decision-transformer python ...
```

Run all experiment commands from the `gym/` directory.

## 3. Datasets

The prepared datasets are stored in `data/`:

- `hopper-medium-replay-v2.pkl`
- `halfcheetah-medium-replay-v2.pkl`
- `ant-medium-replay-v2.pkl`

If the datasets need to be downloaded and prepared again, run:

```bash
env -u PYTHONPATH -u PYTHONHOME MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200" LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:$HOME/.local/share/mamba/envs/decision-transformer/lib:$LD_LIBRARY_PATH" ~/bin/micromamba run -n decision-transformer python data/download_d4rl_datasets.py
```

## 4. Running training

Example: Hopper, Medium-Replay, context length 20, seed 42:

```bash
env -u PYTHONPATH -u PYTHONHOME MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200" LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:$HOME/.local/share/mamba/envs/decision-transformer/lib:$LD_LIBRARY_PATH" ~/bin/micromamba run -n decision-transformer python experiment.py --env hopper --dataset medium-replay --K 20 --seed 42 --device cuda
```

Available environments:

- `hopper`
- `halfcheetah`
- `ant`

Experiment parameters:

| Argument | Meaning |
|---|---|
| `--env` | Environment name |
| `--dataset` | Dataset name (`medium-replay`) |
| `--K` | Context length (1 or 20) |
| `--seed` | Random seed |
| `--device` | Training device (`cuda`) |
| `--max_iters` | Number of training iterations |
| `--num_steps_per_iter` | Gradient updates per iteration |

The default configuration uses 10 iterations with 10,000 updates each, giving 100,000 gradient updates per run.

## 5. Short test run

To check that training works without running a full experiment:

```bash
env -u PYTHONPATH -u PYTHONHOME MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200" LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:$HOME/.local/share/mamba/envs/decision-transformer/lib:$LD_LIBRARY_PATH" ~/bin/micromamba run -n decision-transformer python experiment.py --env hopper --dataset medium-replay --K 20 --seed 42 --max_iters 1 --num_steps_per_iter 20 --num_eval_episodes 1 --device cuda
```

This performs 20 gradient updates and evaluates the model.

## 6. Evaluation and checkpoints

Evaluation is currently integrated into `experiment.py` and runs during training for the configured target returns.

After training, the model weights are saved in:

```text
checkpoints/
```

Example checkpoint:

```text
checkpoints/hopper_medium-replay_K20_seed42.pt
```

The checkpoint can be loaded using PyTorch:

```python
import torch

checkpoint = torch.load(
    "checkpoints/hopper_medium-replay_K20_seed42.pt",
    map_location="cpu"
)
```

## 7. Git

The `.gitignore` file excludes:

```gitignore
checkpoints/
data/*.pkl
```