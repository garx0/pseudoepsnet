# Reproducing Pseudo ε-Net experiments using SAMAROH method
This instruction tells how to reproduce experiments from "Pseudo ε-Net Method with Delayed Decisions Based on Representatives and Anti-Representatives for Multi-Agent Stream Classification" paper.

#### 1. Go to `maroh` directory in the root of this repository

All paths below will be specified relative to this directory, until it is said to go to another directory.

#### 2. Generate traffic matrices for training a SAMAROH model

`python generate_flows.py data_train/abilene_train/train/abilene --seed 1234 --flows 1 --matr 1 --nfiles 120 --intensity 0.2 --bwvar 4 --ecmp --solver --out_path data_train/abilene_train/train/abilene/flows.json`

Move files `flows_100*` -- `flows_119*` from `data_train/abilene_train/train/abilene` to `data_train/abilene_train/test/abilene`.

#### 3. Train a SAMAROH model

`python main.py data_train/abilene_train -t`

Results will appear in new directory `data_train/abilene_train/exp_<name1>`. Take `20000_actor.pt` and `20000_critic.pt` from `data_train/abilene_train/exp_<name1>/results/model` and copy them into `data_train/abilene_train/model/actor.pt` and `data_train/abilene_train/model/critic.pt` respectively.

#### 4. Generate traffic matrices for recording a dataset of states and actions

`python generate_flows.py data_train/abilene_rec/train/abilene --seed 123456 --flows 1 --matr 1 --nfiles 10000 --intensity 0.2 --bwvar 4 --ecmp --solver --out_path data_train/abilene_rec/train/abilene/flows.json`

Remove files from `flows1000*` to `flows7999*` from `data_train/abilene_rec/train/abilene`.
Copy files from `flows0000.json` to `flows0999.json` to `data_train/abilene_clustertrain/train/abilene`.
Copy files from `flows8000*` to `flows9999*` to `data_train/abilene_test/train/abilene` (including .npy and .npz files).
Copy any `flows*.json` file to `data_train/abilene_rec/test/abilene`, `data_train/abilene_test/test/abilene`, `data_train/abilene_clustertrain/test/abilene`.

#### 5. Test SAMAROH model and record resulting states and greedy actions, thus making a dataset for training PεN, KNN, Centroids

`python main.py data_train/abilene_rec -t -m data_train/abilene_train/model`

Results will appear in new directory `data_train/abilene_rec/exp_<name2>`.

In `data_train/abilene_rec/exp_<name2>/results`:

Copy files `states_0-999.npz` and `actions_gr_0-999.npz` to `../representatives/train4_1000 directory`.

Copy files `states_8000-8999.npz`, `states_9000-9999.npz`, `actions_gr_8000-8999.npz, actions_gr_9000-9999.npz` to `../representatives/test4` directory, and rename the files in `representatives/test4` subtracting 8000 from each number (resulting in `*_0-999.npz`, `*_1000-9999.npz`).

#### 6. Go to `representatives` directory in the repository root

All paths below will be specified relative to this directory, until it is said to go to another directory.

#### 7. Train and test PεN, KNN, Centroids classifiers on the recorded dataset

For PεN, run following command for `<value>` = 0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5:
`python main.py --dist-percentile <value>`

For KNN and Centroids, run following commands respectively:

`python main.py --classifier kNN`

`python main.py --classifier centroids`

Results for each experiment will appear in a new subdirectory of results directory. For PεN, it will contain saved representatives, which can be loaded into other experiments.

#### 8. Plot accuracy and coverage of classifiers on test data

List the experiment directories to be plotted in `plot2d_input.json` (see format example in `plot2d_input_example.json`).

Run:

`python plot2d.py -i plot2d_input.json --test test4`

This will reproduce **Figure 1** in the paper.

#### 9. Aggregate and compare results of multiple experiments

Make `table_config.yaml` with paths and names of experiments to analyze, see format example in `table_config_example.yaml`.

Run:

`python table.py -i table_config.yaml`

If run `table.py` on PεN experiment results for dist_percentile values 0.5 and 1, this will reproduce **Table 2** in the paper (need to specify `--no-last-iter` command line argument to reproduce same number of representatives and antirepresentatives however).

#### 10. Go back to `maroh` directory in the root of this repository

#### 11. Test a two-layer method with SAMAROH model and PεN classifier as the experience layer

For each distance percentile value `<value>` in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5] do the following:

- Copy `reprs_*.npz` files from corresponding experiment's directory (located at `../representatives/results/epsilon_net_full_p<value>_...`) into `representatives_p<value>` directory (inside maroh directory);
- Run an experiment:
`python main.py data_train/abilene_test -t -c _p<value> --seed 123 -m data_train/abilene_train/model`
(where `_p<value>` string denotes postfix of existing `config_p<value>.yaml` file in `data_train/abilene_test`).
For convenience, you can specify `-n pen<value>` argument, to include algorithm name and parameter value in experiment results directory name.

#### 12. Test a two-layer method with SAMAROH model and KNN or Centroids classifier as the experience layer

For KNN and Centroids activating on all iterations (`<value>` = 0) or after i-th iteration (`<value>` = i+1, i = 0,1), copy contents of ../representatives/train4_1000 into classifier_data directory (inside `maroh` directory), and run these commands (for KNN and Centroids experiments respectively):

`python main.py data_train/abilene_test -t -c _knn<value> --seed 123 -m data_train/abilene_train/model`

`python main.py data_train/abilene_test -t -c _centroids<value> --seed 123 -m data_train/abilene_train/model`

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
    
Make `plot_advanced_input.json` file with paths to experiment results directories from `data_train/abilene_rec` (SAMAROH Φ values without experience layer) and `data_train/abilene_test`, see format example in `plot_advanced_input_example.json` example. Also specify `data_train/abilene_test/train/abilene` path there, which contains .npz files with ECMP and LP-solver values calculated for the same traffic matrices.

This will reproduce **Table 1** and **Figures 2 and 3** in the paper.

#### 16. Update trained PεN classifier by steps to measure update time

For each `<dp>` in [0.5, 1], for each `<len>` in [4, 8, 16, 32, 64, 128, 256], run:

`python main.py --config config_update.yaml --dist-percentile <dp> --reprs-path results/epsilon_net_full_p<dp>... --update --update-buffer-length <len> --max-updates 10`

(while specifying `--reprs-path` as directory with representatives from PεN training experiment with distance percentile `<dp>`).

#### 17. Aggregate update time experiment results into a table

Make `update_time_analysis_input.json` file with directories and names of update time measurement experiments, see format example in `update_time_analysis_input_example.json`.

Run:

`python update_time_analysis.py -i update_time_analysis_input.json`

This will reproduce **Table 3** in the paper.

