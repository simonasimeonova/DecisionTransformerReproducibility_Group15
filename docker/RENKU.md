# Training on Renku

The existing training image supplies the dependencies. A Job can download
`odei/simple-training` and run this branch's `docker/train.sh` without rebuilding
the image. The script then uses this branch's `gym/experiment.py`.

1. Ensure the project data connector is mounted at `DecisionTransformerReproducibility_Group15/training-storage`, with `.pkl` files in its `datasets/` folder.
2. In the Job command, download the branch archive into the repository folder, then run:

   ```bash
   bash docker/train.sh ant medium-replay 20 42
   ```

   The four values are environment, dataset, K, and training seed. To train three seeds in one Job, use `42,43,44` as the last value. Optional `experiment.py` arguments can follow them.
3. Select a GPU resource. Seeds run sequentially, each with its own archive in `training-storage`. The existing evaluation script reads those archives.

The wrapper links the dataset to the `data/` path expected by `experiment.py`, runs training, archives the checkpoint and log, and waits five minutes for SWITCHdrive upload. It does not download datasets. For a short test, append `--max_iters 1 --num_steps_per_iter 20 --num_eval_episodes 1`.

If you later build a new image from this branch, `/usr/local/bin/train` is the
same script and uses the code baked into that image.

Outside this Job wrapper, `experiment.py` also accepts `--seeds 42 43 44` directly. It trains them sequentially and saves three distinct checkpoint files in the current `checkpoints/` directory.
