# Bayesian Continuous

Benchmarking continuous optimization techniques for DAG (Bayesian network) structure
learning against classical discrete search algorithms.

We compare **NOTEARS** and **DAG-GNN** (continuous optimization) against **PC**, **FGES**, and **Hill Climbing** (classical search) on synthetic networks from the [bnlearn repository](https://www.bnlearn.com/bnrepository/) (Cancer, Child, Alarm) and the real-world [Sachs et al. (2005)](https://zenodo.org/records/7681811) protein signaling dataset. Each algorithm is scored on Structural Hamming Distance (SHD) against the ground-truth DAG and on execution time.

See [report/main.pdf](report/main.pdf) for the full write-up of the method and results.

## Project structure

- `data.py` dataset loading/generation
- `training.py` training entrypoints for each algorithm and the grid-search routine used for hyperparameter tuning
- `reporting.py` graph reconstruction metrics and plotting helpers
- `gridsearch.ipynb` hyperparameter grid search per algorithm/dataset
- `experiment.ipynb` final experiment: runs every algorithm on every dataset with
  the tuned hyperparameters and produces the results reported
- `datasets/` bnlearn `.bif` networks and the Sachs et al. dataset (not tracked in git)
- `report/` LaTeX source and build script for the report

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Datasets

`datasets/` is gitignored and must be populated manually:

- **bnlearn networks** (Cancer, Child, Alarm, Barley): download the `.bif` files from the [bnlearn repository](https://www.bnlearn.com/bnrepository/) into `datasets/bnlearn/`.
- **Sachs et al.**: download from [Zenodo](https://zenodo.org/records/7681811) into `datasets/sachs/`, keeping the `Data Files/` directory and `GroundTruth.csv`.
