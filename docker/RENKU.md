# Training on Renku

The Dockerfile builds one image with the dependencies and `experiment.py`.
The same image can train any supported environment, dataset, K, or seed.

1. Build this branch's image with the manual GitHub Actions workflow. Use its commit-SHA image tag in a Renku Job launcher.
2. Ensure the project data connector is mounted at `DecisionTransformerReproducibility_Group15/training-storage`, with `.pkl` files in its `datasets/` folder.
3. Set the Job command to a JSON array such as:

   ```json
   ["/usr/local/bin/train", "ant", "medium-replay", "20", "42"]
   ```

   The four values are environment, dataset, K, and training seed. To train three seeds in one Job, use `"42,43,44"` as the last value. Optional `experiment.py` arguments can follow them.
4. Select a GPU resource. Seeds run sequentially, each with its own archive in `training-storage`. The existing evaluation script reads those archives.

The wrapper links the dataset to the `data/` path expected by `experiment.py`, runs training, archives the checkpoint and log, and waits five minutes for SWITCHdrive upload. It does not download datasets. For a short test, append `"--max_iters", "1", "--num_steps_per_iter", "20", "--num_eval_episodes", "1"` to the command array.

Outside this Job wrapper, `experiment.py` also accepts `--seeds 42 43 44` directly. It trains them sequentially and saves three distinct checkpoint files in the current `checkpoints/` directory.
