# Evaluation settings — step 1

`load_settings.py` reads two YAML files and returns one dictionary:

1. `configs/defaults.yaml`: shared evaluation settings.
2. An experiment YAML: checkpoints and targets for one environment–dataset pair.

Experiment settings override defaults. Lists are replaced, not combined.
Checkpoint paths are relative to the experiment YAML file.

From the repository's `gym` directory, preview the combined settings:

```bash
python -m evaluation_2.load_settings evaluation_2/configs/hopper_medium_replay.yaml
```

Requires PyYAML. This only loads settings; it does not validate experiments or run
evaluation. Targets and checkpoint paths are still undecided. Targets will be in
raw reward units; the ten default seeds are pilot settings.

`run.py` currently previews settings only. Checkpoint loading, simulator rollouts,
and result saving are not implemented yet. The training runner currently saves
raw weights, so architecture and normalization will need to be supplied separately.
