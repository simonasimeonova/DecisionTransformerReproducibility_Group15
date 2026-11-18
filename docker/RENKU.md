# Ant K=1 (default seed=42) on Renku Jobs

The Dockerfile includes MuJoCo 2.0, its public legacy activation key, system
OpenGL/OSMesa libraries, Python 3.8, Gym 0.18.3, mujoco-py 2.0.2.13,
PyTorch 1.8.1 with CUDA 11.1 (A100 support), Transformers 4.5.1 and W&B 0.9.1.
It compiles and tests Ant and imports the actual DT model during the build,
including a second check as the non-root runtime user. GPU execution is checked
when the Job starts. CPU-only rendering does not disable GPU model training.

## 1. Build an image without Docker in the Renku Session

Commit and push Dockerfile, .dockerignore, docker/, and .github/workflows/docker.yml
to the GitHub repository's default branch. The workflow is MANUAL: merely pushing
does not build/publish an image.

In GitHub: Actions > Build Renku training image > Run workflow.
The workflow builds linux/amd64 and pushes to GitHub Container Registry (GHCR)
using the built-in GITHUB_TOKEN; no additional registry password is needed.
You need repository permissions to run Actions and publish packages.

For the current repository, the image will be:

```text
ghcr.io/simonasimeonova/decisiontransformerreproducibility_group15:ant-k1
```

After a successful build, open the GitHub account/organization's Packages page,
select the package, and ensure Renku can pull it. The simplest option is to make
the image public (Package settings > Change visibility). A public repository
does not automatically make a newly published GHCR package public. If the image
must stay private, configure registry credentials in Renku instead.

For a reproducible run, use the commit SHA image tag from the workflow instead
of the mutable ant-k1 tag. The code baked into that image is what runs;
editing a linked Renku repository does not change it.

## 2. Renku Job Launcher

- Environment: **External environment** (not Create from code).
- Image: the GHCR image above, after its build succeeds.
- Job command: `["/usr/local/bin/train-ant"]`
- Job args: leave empty for the full 100,000-update training run.
- Resources: one GPU allocation (the ~20 GB A100 MIG allocation is sufficient
  for the configured small DT model; actual availability is checked at startup).
- Advanced settings, if requested: UID `1000`, GID `1000`, mount directory
  `/home/renku/work`, working directory `/home/renku/work`.
- Mount writable persistent storage at `/home/renku/work` using your project's
  available storage configuration. Do not assume a Job shares a Session's disk:
  explicitly check the Job's storage settings and output retention.

For a short test, Job args (JSON array):

```json
["--max_iters", "1", "--num_steps_per_iter", "20", "--num_eval_episodes", "1"]
```

After it succeeds, clear Job args for the full run. Each invocation creates a
new output directory, so the short test cannot overwrite a full-run checkpoint.

## 3. Data and outputs

On first launch, the script downloads **only ant-medium-replay-v2** from the
official D4RL dataset URL and converts it with the repository's trajectory
splitting rules. No installation/import of unrelated D4RL environments is needed.
The data is cached in `/home/renku/work/datasets/` for later runs if that storage
is retained. Downloading requires outbound access to rail.eecs.berkeley.edu.
If the host is unavailable, obtain the same Ant v2 HDF5 separately and place it
at `datasets/ant_medium_replay-v2.hdf5`, or supply the already prepared
`datasets/ant-medium-replay-v2.pkl` from your group's trusted source. Do not
substitute AntMaze, another dataset version, or another environment.

Outputs are in `/home/renku/work/results/ant_K1_seed42_<time>_<unique>/`:

- `train.log`: training progress and integrated evaluation results.
- `arguments.json`, `packages.txt`, `source-commit.txt`, `experiment-source.py`:
  configuration, installed versions and code provenance.
- `checkpoints/ant_medium-replay_K1_seed42.pt`: final model weights.

The existing training script saves only at the END and has no resume support.
A stopped/preempted Job loses unfinished training progress. Ant target returns
remain the current code's `[3000, 1500]` (marked TODO upstream); this image does
not implement D's target sweep or normalized-score analysis.

## Optional local build (Docker Desktop / Linux Docker)

From the repository root, with the Docker daemon running:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t dt-ant:local --load .
```

This runs CPU build checks, but does not prove GPU availability. On a Linux
NVIDIA Docker host, test the GPU and persistent outputs with:

```bash
mkdir -p outputs
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/outputs:/home/renku/work" dt-ant:local \
  /usr/local/bin/train-ant \
  --max_iters 1 --num_steps_per_iter 20 --num_eval_episodes 1
```

## Persistent output launcher (recommended)

The old `ant-k1-seed3` image hard-coded seed 3. Rebuild and use the new
`ant-k1` tag (or its new commit SHA tag). The fixed launcher defaults to the
training code's seed 42 and accepts explicit `--seed` overrides. The repository
does not prescribe a list of three seeds; agree that list with the K=20 runs.

For the first full run:

- Command: `["/usr/local/bin/train-ant-persistent"]`
- Arguments: `["--seed", "42", "--max_iters", "10", "--num_steps_per_iter", "10000", "--num_eval_episodes", "100"]`

The persistent wrapper checks the existing Connector mount at
`/home/renku/work/DecisionTransformerReproducibility_Group15/training-storage`,
trains locally, archives the model/configuration/logs to that mount, logs disk
usage and archive size, and waits 300 seconds before exiting. The archive name
contains the actual seed. Check SWITCHdrive directly and download the archive;
mount readback and a delay cannot prove remote upload completed. Override the
mount path with `DT_OUTPUT_DIR` and the wait with `DT_UPLOAD_WAIT_SECONDS` if
needed. Outputs are copied only after the training process exits; interruption
can still lose unfinished progress.
