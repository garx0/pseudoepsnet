import json
import numpy as np
import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read two files via named parameters')
    parser.add_argument('-i', '--input', default='update_time_analysis_input.json', help='Path to json file with experiment paths')
    parser.add_argument('-o', '--out', default='update_time_analysis_table', help='File name prefix for output table')
    args = parser.parse_args()
    paths_json_filepath = args.input
    with open(paths_json_filepath, 'r', encoding='utf-8') as f:
        paths_json_data = json.load(f)
    exp_dirs_names = paths_json_data["exp_dirs_names"]
    n_agents = int(paths_json_data["n_agents"])
    n_msg_iter = int(paths_json_data["n_iterations"])
    out_filename = args.out
    out_filename_noext, out_filename_ext = os.path.splitext(out_filename)

    durations_results = []
    columns = ["experiment", "mean duration, μs", "duration std, μs"]
    for exp_dir, exp_name in exp_dirs_names:
        durations = []
        for agent in range(n_agents):
            for i in range(n_msg_iter):
                data = np.load(os.path.join(exp_dir, f"updates_a{agent:02}_i{i}.npy"))
                durations.append(np.mean(data))
        durations = np.array(durations)
        mean_dur_us = np.mean(durations*1000000)
        std_dur_us = np.std(durations*1000000, ddof=1)
        durations_results.append((exp_name, mean_dur_us, std_dur_us))
        print(f"{exp_name}: duration = {mean_dur_us:.2f} +- {std_dur_us:.2f} us")
    df = pd.DataFrame(durations_results, columns=columns)
    out_filename_actual = f"{out_filename_noext}.csv"
    df.to_csv(out_filename_actual)
    print(f"table saved to {out_filename_actual}")
