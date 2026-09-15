# Prototype-based baselines (LVQ1 / GLVQ / sklearn-GLVQ / LRVQ / Prototypical Net)

This directory reproduces the comparison of Pseudo ε-Net with published external
prototype-based classifiers that is reported in the paper
"Pseudo ε-Net Method with Delayed Decisions Based on Representatives and
Anti-Representatives for Multi-Agent Stream Classification"
(Figure 3 and Table A1 for Abilene, and Figure 4 and Table A2 for Geant2009, and the
corresponding paragraph in the Discussion section).

## Methods (all implemented in `baselines.py`)

| Key             | Method                             | Reference                        | Reject option |
|-----------------|------------------------------------|----------------------------------|---------------|
| `LVQ1`          | Learning Vector Quantization (LVQ1)| Kohonen 1990                     | no            |
| `GLVQ`          | Generalized LVQ                    | Sato & Yamada 1995               | no            |
| `GLVQ_sklearn`  | Open-source GLVQ (sklearn-lvq)     | Biehl et al., sklearn-lvq        | no            |
| `LRVQ`          | LVQ + local reject option          | Fischer et al. 2016              | yes           |
| `ProtoNet`      | Prototypical Network (linear emb.) | Snell et al. 2017                | no            |

All classifiers follow the repository metric convention:
accuracy = correct confident predictions / confident predictions,
coverage  = confident predictions / total test points,
compression = number of stored representatives / training set size.

## Requirements

Installed on top of `../requirements.txt`:

- `scikit-learn` (`sklearn-lvq`) — needed only for `GLVQ_sklearn`
- `torch` — needed only for `ProtoNet`

## Data

Download the train/test datasets from the Zenodo data repository
(DOI 10.5281/zenodo.21059817) and unpack them into representatives directory, so
that it contains:

    train_abilene/    test_abilene/     # for Abilene
    train_geant2009/  test_geant2009/   # for Geant2009

Specify train and test dataset paths (relative to `representatives/baselines` directory) in a `<config_name>.json` file (example in config_abilene_example.json) and run `python run_baselines.py -i <config_name>.json`.

## Running the experiments

From this directory, run:

    python run_baselines.py    -i config_abilene_example.json
    python run_baselines.py    -i config_geant2009_example.json

Each run is checkpointed after every model: re-launching the same command
resumes, skipping already-completed models. When all models are done, an
`analysis_*.json` is written into `./results/<out_dir>/`.

## Results used in the paper

The analysis files that produced corresponding rows of Tables 2 and 3 are
committed in `./results/` in Zenodo data repository (DOI 10.5281/zenodo.21059817):

    results/baselines_train_abilene_20260813_135019/analysis_20260813_135019.json
    results/baselines_train_geant2009_20260813_125814/analysis_20260813_125814.json

To regenerate the per-iteration tables from committed results, go to `representatives` directory and run `python plot2d.py -i config_abilene.json` with specifying experiment result path in `representatives/config_abilene.json` file (see example in `representatives/config_abilene_example.json`, which already contains paths to experiment results which are available in Zenodo data repository).
