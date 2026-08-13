# Prototype-based baselines (LVQ1 / GLVQ / sklearn-GLVQ / LRVQ / Prototypical Net)

This directory reproduces the comparison of P ε-Net with published external
prototype-based classifiers that is reported in the paper
"Pseudo ε-Net Method with Delayed Decisions Based on Representatives and
Anti-Representatives for Multi-Agent Stream Classification"
(Tables `tab:perf` for Abilene and `tab:perf_geant` for Geant2009, and the
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

`sklearn-lvq` requires a small compatibility shim (already applied inside
`SklearnGLVQ`) for recent scipy (1-D `x0` in `scipy.optimize.minimize`) and
numpy 2.x (`np.math` removal).

## Data

Download the train/test data sets from the Zenodo data repository
(DOI 10.5281/zenodo.21672588) and unpack them into a single directory, so
that it contains:

    train_abilene/    test_abilene/     # for Abilene
    train_geant2009/  test_geant2009/   # for Geant2009

Pass this directory to the run scripts below with `--data <dir>` (or export
`BASELINES_DATA`).

## Running the experiments

From this directory:

    python run_baselines_abilene.py    --data <dir>   # 20 agents x it.0..it.3 x 5 models
    python run_baselines_geant2009.py  --data <dir>   # 78 agents x it.0..it.5 x 5 models

Each run is checkpointed after every model: re-launching the same command
resumes, skipping already-completed models. When all models are done, an
`analysis_*.json` is written into `./results/<run_id>/`.

## Results used in the paper

The analysis files that produced Tables `tab:perf` / `tab:perf_geant` are
committed in `./results/`:

    results/baselines_abilene_20260813/analysis_20260813_135019.json
    results/baselines_geant2009_20260813/analysis_20260813_125814.json

To regenerate the per-iteration and overall tables from committed (or just
re-run) results:

    python summarize_baselines.py [ablene_analysis.json [geant_analysis.json]]