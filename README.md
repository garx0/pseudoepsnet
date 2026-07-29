# Reproducing Pseudo ε-Net experiments using SAMAROH method
This instruction tells how to reproduce experiments from "Pseudo ε-Net Method with Delayed Decisions Based on Representatives and Anti-Representatives for Multi-Agent Stream Classification" paper.

### A. Experiments on Abilene topology

#### 1. Go to `maroh` directory in the root of this repository

All paths below will be specified relative to this directory, until it is said to go to another directory.

#### 2. Generate traffic matrices for training a SAMAROH model

`python generate_flows.py data_train/abilene_train/train/abilene --seed 1234 --flows 1 --matr 1 --nfiles 120 --intensity 0.2 --bwvar 4 --ecmp --solver --out_path data_train/abilene_train/train/abilene/flows.json`

Move files `flows_100*` -- `flows_119*` from `data_train/abilene_train/train/abilene` to `data_train/abilene_train/test/abilene`.

#### 3. Train a SAMAROH model

`python main.py data_train/abilene_train -t`

Results will appear in new directory `data_train/abilene_train/exp_<name1>`. Take `20000_actor.pt` and `20000_critic.pt` (corresponding to last checkpoint, since training takes 20000 episodes in this case) from `data_train/abilene_train/exp_<name1>/results/model` and copy them into `data_train/abilene_train/model/actor.pt` and `data_train/abilene_train/model/critic.pt` respectively.

#### 4. Generate traffic matrices for recording a dataset of states and actions

`python generate_flows.py data_train/abilene_rec/train/abilene --seed 123456 --flows 1 --matr 1 --nfiles 10000 --intensity 0.2 --bwvar 4 --ecmp --solver --out_path data_train/abilene_rec/train/abilene/flows.json`

Remove files from `flows1000*` to `flows7999*` from `data_train/abilene_rec/train/abilene`.

Copy files from `flows0000.json` to `flows0999.json` to `data_train/abilene_clustertrain/train/abilene`.

Copy files from `flows8000*` to `flows9999*` to `data_train/abilene_test/train/abilene` (including .npy and .npz files).

Copy any single `flows*.json` file to `data_train/abilene_rec/test/abilene`, `data_train/abilene_test/test/abilene`, `data_train/abilene_clustertrain/test/abilene` (these flows are not used for experimental results, but MAROH will not run if there are no flows in these directories).

To clarify, naming of `data_train/*/train` and `data_train/*/test` directories corresponds to MAROH training mode. However, experiments in abilene_rec, abilene_test, abilene_clustertrain directories run in training mode but with actual training turned off, so in this case it is just evaluation of MAROH on flows in train directory, while results on flows in test directory are ignored.

If reproducing exact same flows as in paper's experiments is not required, this procedure can be simplified by generating only 3000 flows (`--nfiles 3000`) and copying files corresponding to last 2000 flows to `data_train/abilene_test/train/abilene`.

In general, to generate N_train and N_test flow matrices, use `--nfiles <N_train+N_test>` and copy json files for first N_train flow matrices to `data_train/abilene_clustertrain/train/abilene` and all files for last N_test flow matrices to `data_train/abilene_test/train/abilene`.

#### 5. Test SAMAROH model and record resulting states and greedy actions, thus making a dataset for training PεN, k-NN, Centroids

`python main.py data_train/abilene_rec -t -m data_train/abilene_train/model`

Results will appear in new directory `data_train/abilene_rec/exp_<name2>`.

In `data_train/abilene_rec/exp_<name2>/results`:

Copy files `states_0-999.npz` and `actions_gr_0-999.npz` to `../representatives/train_abilene directory`.

Copy files `states_1000-1999.npz`, `states_2000-2999.npz`, `actions_gr_1000-1999.npz, actions_gr_2000-2999.npz` to `../representatives/test_abilene` directory, and rename the files in `representatives/test_abilene` subtracting 1000 from each number, so that the numbering in `representatives/test_abilene` starts from 0 (resulting in `*_0-999.npz`, `*_1000-9999.npz`).

#### 6. Go to `representatives` directory in the repository root

All paths below will be specified relative to this directory, until it is said to go to another directory.

#### 7. Train and test PεN, k-NN, Centroids classifiers on the recorded dataset

For PεN, run following command for `<value>` in [0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]:

`python main.py --config config_abilene.yaml --dist-percentile <value>`

To test PεN without antirepresentatives feature, run:

`python main.py --config config_abilene.yaml --dist-percentile <value> --disable-antireprs`

For k-NN and Centroids, run following commands respectively:

`python main.py --config config_abilene.yaml --classifier kNN`

`python main.py --config config_abilene.yaml --classifier centroids`

Results for each experiment will appear in a new subdirectory of results directory. For PεN, it will contain saved representatives, which can be loaded into other experiments.

#### 8. Plot accuracy and coverage of classifiers on test data

List the experiment directories to be plotted in `plot2d_input.json` (see format example in `plot2d_input_abilene_example.json`).

If experiments for plotting only feature PεN (with antirepresentatives), k-NN, Centroids experiments, run:

`python plot2d.py -i plot2d_input.json`

If experiments for plotting contain PεN without antirepresentatives, run:

`python plot2d_abtests.py -i plot2d_input.json`

This will reproduce **Figure 3** in the paper.

#### 9. Aggregate and compare results of multiple experiments

Make `table_config.yaml` with paths and names of experiments to analyze, see format example in `table_config_abilene_example.yaml`.

Run:

`python table.py -i table_config.yaml`

If run `table.py` on PεN experiment results for dist_percentile values 0.5 and 1, this will reproduce **Table 6** in the paper (need to specify `--no-last-iter` command line argument to reproduce same number of representatives and antirepresentatives however).

#### 10. Go back to `maroh` directory in the root of this repository

#### 11. Test a two-layer method with SAMAROH model and PεN classifier as the experience layer

For each distance percentile value `<value>` in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10] do the following:

* Copy `reprs_*.npz` files from corresponding experiment's directory (located at `../representatives/results/epsilon_net_full_p<value>_...`) into `representatives_abilene_p<value>` directory (inside maroh directory);

* Run an experiment:
`python main.py data_train/abilene_test -t -c _p<value> --seed 123 -m data_train/abilene_train/model`

(where `_p<value>` string denotes postfix of existing `config_p<value>.yaml` file in `data_train/abilene_test`).

For convenience, you can specify `-n pen<value>` argument, to include algorithm name and parameter value in experiment results directory name.

Note that in `python main.py <train_dir> -t -c <name> ...` commands, `-c` argument does not specify any parameter directly, but rather specifies the `<train_dir>/config<name>.yaml` configuration file to be loaded, therefore that file should exist beforehand. To run experiments on parameter values outside the paper's experimental research scope, modify contents of an existing configuration file (where all parameter values are specified).

#### 12. Test a two-layer method with SAMAROH model and k-NN or Centroids classifier as the experience layer

For k-NN and Centroids activating on all iterations (`<value>` = 0) or after i-th iteration (`<value>` = i+1, i = 0..K-2, where K=3 for Abilene experiments), run these commands (for k-NN and Centroids experiments respectively):

`python main.py data_train/abilene_test -t -c _knn<value> --seed 123 -m data_train/abilene_train/model`

`python main.py data_train/abilene_test -t -c _centroids<value> --seed 123 -m data_train/abilene_train/model`

These experiments will use training data from path `../representatives/train_abilene` relative to `maroh` directory root, so make sure that `maroh` and `representatives` directories are located in same directory.

For convenience, you can specify `-n knn<value>` or `-n centroids<value>` argument, to include algorithm name and parameter value in experiment results directory name.

#### 13. Prepare experience layer for testing SAMAROH-2L method (with clustering)

For each Δ value `<value>` in [0.003, 0.004], run:

`python main.py data_train/abilene_clustertrain -t -c _<value> --seed 123 -m data_train/abilene_train/model`

Result for each `<value>` will appear in new directory `data_train/abilene_clustertrain/<name4_value>`. Representatives will be saved in `data_train/abilene_clustertrain/<name4_value>/results/model/memory.npz`

For convenience, you can specify `-n cluster<value>` argument, to include algorithm name and parameter value in experiment results directory name.

#### 14. Test SAMAROH-2L method

For each Δ value `<value>` in [0.003, 0.004], run:

`python main.py data_train/abilene_test -t -c _cluster_<value> --seed 123 -m data_train/abilene_train/model --memory data_train/abilene_clustertrain/<name4_value>/results/model/memory.npz`

For convenience, you can specify `-n cluster<value>` argument, to include algorithm name and parameter value in experiment results directory name.

#### 15. Make plot and tables of two-layer method experiments
    
Make `plot_advanced_input.json` file with paths to experiment results directories from `data_train/abilene_rec` (SAMAROH Φ values without experience layer) and `data_train/abilene_test`, see format example in `plot_advanced_input_abilene_example.json` example. Also specify `data_train/abilene_test/train/abilene` path there, which contains .npz files with ECMP and LP-solver values calculated for the same traffic matrices. Run:

`python plot_advanced.py -i plot_advanced_input.json`

This will reproduce **Tables 2-3** (but for single experimental run instead of 12) and **Figures 5 and 6** in the paper.

To reproduce **Tables 2-3** in the paper exactly, run steps 1-15 12 times independently, but generating N_train=1000 and N_test=1000 (instead of 2000) flow matrices at step 4, and specifying `--seed` in each run with a value from [151, 152, 161, 162, 171, 172, 181, 182, 191, 192, 201, 202] in each command where `--seed` is specified. Then concatenate tables from 12 .csv files produced by `plot_advanced.py` for each run, and calculate mean and standard deviation (ddof=1) for groups of rows sharing same value of `classifier` column.

#### 16. Update trained PεN classifier by steps to measure update time

For each `<dp>` in [0.5, 1], for each `<len>` in [4, 8, 16, 32, 64, 128, 256], run:

`python main.py --config config_update.yaml --dist-percentile <dp> --reprs-path results/epsilon_net_full_p<dp>... --update --update-buffer-length <len> --max-updates 10`

(while specifying `--reprs-path` as directory with representatives from PεN training experiment with distance percentile `<dp>`).

#### 17. Aggregate update time experiment results into a table

Make `update_time_analysis_input.json` file with directories and names of update time measurement experiments, see format example in `update_time_analysis_input_abilene_example.json`.

Run:

`python update_time_analysis.py -i update_time_analysis_input.json`

This will reproduce **Table 7** in the paper.

### B. Experiments on Geant2009 topology

The procedure is same as for Abilene topology (with difference in parameter values and directory paths), except that some experiments were not conducted on Geant2009 topology.

Use Abilene instructions with the following differences:
* Replace "abilene" to "geant2009" in every instruction.
* Command for generating MAROH training flows dataset (step 2) is:

`python generate_flows.py data_train/geant2009_train/train/geant2009 --seed 1234 --flows 1 --matr 1 --nfiles 120 --intensity 0.125 --bwvar 4 --ecmp --solver --out_path data_train/geant2009_train/train/geant2009/flows.json`

* Number of matrices in training and testing datasets for classifiers (generated at steps 4, 5) are N_train=500 and N_test=500 respectively (while for Abilene they were N_train=1000 and N_test=2000). Command for generating flows dataset for training and testing classifiers (step 4) is:

`python generate_flows.py data_train/geant2009_rec/train/geant2009 --seed 12345 --flows 1 --matr 1 --nfiles 1000 --intensity 0.125 --bwvar 4 --ecmp --solver --out_path data_train/geant2009_rec/train/geant2009/flows.json`

* File copying procedure after running command in step 4 is the following:

    * Copy files from `flows0000.json` to `flows0499.json` to `data_train/geant2009_clustertrain/train/geant2009`;

    * Copy files from `flows0500*` to `flows0999*` to `data_train/geant2009_test/train/geant2009` (including .npy and .npz files);

    * Copy any `flows*.json` file to `data_train/geant2009_rec/test/geant2009`, `data_train/geant2009_test/test/geant2009`, `data_train/geant2009_clustertrain/test/geant2009`.

* Geant2009 experiments used different values of PεN's distance percentile and SAMAROH-2L's Δ parameter, they can be found in the paper (or in config file names in `data_train/geant2009_test`, `data_train/geant2009_clustertrain`). For step 12, note that Geant2009 experiments used K=5 (number of message iterations).

* Experiments at steps 16-17 were not conducted on Geant2009 topology in the paper.




