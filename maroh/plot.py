import sys
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def find_convergence(phi_values):
    window = 100
    C = 0.01
    i = 0
    for i in range(int(len(phi_values) / window)):
        end = min((i + 1) * window, len(phi_values))
        phi_avg = sum(phi_values[i * window:end]) / window
        if phi_avg < C:
            return end
    return len(phi_values)


class ResultFolder:
    def __init__(self, path, parameter_name='phi_values'):
        self._folder_path = path
        self._results = defaultdict(lambda: [])
        self._results2 = []
        self._max_episode = 0
        self._parameter_name = parameter_name
        self._parse_folder()

    def _parse_file(self, file_obj, start):
        data_list = json.load(file_obj)
        episode_index = start
        for episode_data in data_list:
            self._results[episode_index] = episode_data
            episode_index += 1

        if episode_index > self._max_episode:
            self._max_episode = episode_index

    def _parse_folder(self, path=None):
        cur_path = path if path else self._folder_path
        with os.scandir(cur_path) as files:
            for file in files:
                if file.is_dir(follow_symlinks=False):
                    self._parse_folder(file.path)
                    continue
                if not file.is_file():
                    continue
                match = re.search(f'({self._parameter_name})_([0-9]+)-([0-9]+).json', file.name)
                if not match:
                    continue
                with open(file.path, 'r') as f:
                    self._parse_file(f, int(match[2]))
                print(f'parsed file {file.name}')

    def _parse_file2(self, file_obj):
        data_list = json.load(file_obj)
        for run_data in data_list:
            self._results2.append(run_data[0])

    def _parse_folder2(self, path=None):
        cur_path = path if path else self._folder_path
        with os.scandir(cur_path) as files:
            for file in files:
                if not file.is_file():
                    continue
                match = re.search('avg.json', file.name)
                if not match:
                    continue
                with open(file.path, 'r') as f:
                    self._parse_file2(f)
                print(f'parsed file {file.name}')

    def _plot(self, values: list, start: int, end: int, amount: int, x_name: str, filename: str):
        plt.rcParams.update({'font.size': 22})
        # plt.rc('xtick', labelsize=14)

        np_phi_values = np.array(values)
        np_linear = np.linspace(start, end, amount)
        pixel = 1/plt.rcParams['figure.dpi']
        figure, ax = plt.subplots(figsize=(1200*pixel, 800*pixel))
        ax.plot(np_linear, np_phi_values, label='phi')
        ax.set_xlabel(x_name)
        ax.set_ylabel(self._parameter_name)
        # ax.legend()
        # plt.ylim(0.01, 0.25)
        # plt.yticks([0.02, 0.022, 0.024, 0.026, ])
        plt.savefig(os.path.join(self._folder_path, filename), bbox_inches='tight')
        plt.close()

    def _boxplot(self, values: list, episodes_per_matrix: int, x_name: str, filename: str):
        box_values = []
        for matrix_num in range(0, len(values) // episodes_per_matrix):
            box_values.append(np.array(values[matrix_num * episodes_per_matrix:(matrix_num + 1) * episodes_per_matrix]))
        np_phi_values = np.transpose(np.array(box_values))
        pixel = 1/plt.rcParams['figure.dpi']
        figure, ax = plt.subplots(figsize=(1900*pixel, 800*pixel))
        ax.boxplot(np_phi_values)
        # ax.set_xlabel(x_name)
        # ax.set_ylabel('phi value')
        # ax.legend()
        plt.savefig(os.path.join(self._folder_path, filename), bbox_inches='tight')
        plt.close()

    def prepare_phi_values(self, max_episode=None):
        if max_episode:
            self._max_episode = max_episode

        # phi_values = [min(self._results[episode]) for episode in range(max_episode)]
        phi_values = [self._results[episode][-1] for episode in range(self._max_episode)]

        return phi_values

    def prepare_averaged_phi_values(self):
        # max_episode = min(self._max_episode, 1000)
        # phi_values = [min(self._results[episode]) for episode in range(max_episode)]
        plot_period = 5

        phi_values = [self._results[episode][-1] for episode in range(self._max_episode)]

        averaged_values = []
        number_of_points = (len(phi_values) // plot_period)
        if len(phi_values) % plot_period != 0:
            number_of_points += 1

        for i in range(number_of_points):
            start = i * plot_period
            end = min((i + 1) * plot_period, len(phi_values))
            averaged_values.append(sum(phi_values[start:end]) / (end - start))

        return averaged_values

    def plot_average_and_full(self, plot_period, parameter_name='phi_values', max_episode=None):
        if max_episode:
            self._max_episode = max_episode

        phi_values = [min(self._results[episode]) for episode in range(self._max_episode)]
        # phi_values = [self._results[episode][-1] for episode in range(self._max_episode)]

        averaged_values = []
        number_of_points = (len(phi_values) // plot_period)
        if len(phi_values) % plot_period != 0:
            number_of_points += 1

        for i in range(number_of_points):
            start = i * plot_period
            end = min((i + 1) * plot_period, len(phi_values))
            averaged_values.append(sum(phi_values[start:end]) / (end - start))

        # full graph
        self._plot(phi_values, 0, len(phi_values) - 1, len(phi_values), 'episode', f'{self._parameter_name}_0-{len(phi_values)}.png')

        # averaged graph
        self._plot(averaged_values, 0, len(averaged_values) - 1, len(averaged_values),
                   f'episode batch ({plot_period} per batch)', f'{self._parameter_name}_averaged_full.png')

def plot_multi(values_multi: list[list], legend: list[str], start: int, end: int, amount: int, x_name: str,
               filename: str, folder_path: str):
    plt.rcParams.update({'font.size': 22})
    # plt.rc('xtick', labelsize=14)

    np_linear = np.linspace(start, end, amount)
    pixel = 1/plt.rcParams['figure.dpi']
    figure, ax = plt.subplots(figsize=(1200*pixel, 800*pixel))
    for values, label in zip(values_multi, legend):
        np_values = np.array(values)
        # print(f"shape: {np_values.shape} -- {label}")
        ax.plot(np_linear, np_values, label=label)
    ax.set_xlabel(x_name)
    ax.legend(fontsize=9)
    # plt.ylim(0.01, 0.25)
    # plt.yticks([0.02, 0.022, 0.024, 0.026, ])
    plt.savefig(os.path.join(folder_path, filename), bbox_inches='tight')
    plt.close()

def plot_average_and_full_multi(plot_period, folders, max_episode=None):
    if max_episode is None:
        max_episode = folders[0]._max_episode
        assert(all(folder._max_episode == folders[0]._max_episode for folder in folders))

    folder_path = folders[0]._folder_path
    assert(all(folder._folder_path == folders[0]._folder_path for folder in folders))

    values_multi = [[min(folder._results[episode]) for episode in range(max_episode)] for folder in folders]
    # values_multi = [[folder._results[episode][-1] for episode in range(max_episode)] for folder in folders]

    averaged_values_multi = []
    for values in values_multi:
        averaged_values = []
        number_of_points = (len(values) // plot_period)
        if len(values) % plot_period != 0:
            number_of_points += 1
        averaged_values_multi.append(averaged_values)

    for values, averaged_values in zip(values_multi, averaged_values_multi):
        for i in range(number_of_points):
            start = i * plot_period
            end = min((i + 1) * plot_period, len(values))
            averaged_values.append(sum(values[start:end]) / (end - start))

    N = len(values_multi[0])
    # print(f"lenghts: {[len(values) for values in values_multi]}")
    assert(all(len(values) == len(values_multi[0]) for values in values_multi))

    N_averaged = len(averaged_values_multi[0])
    # print(f"lenghts: {[len(averaged_values) for averaged_values in averaged_values_multi]}")
    assert(all(len(averaged_values) == len(averaged_values_multi[0]) for averaged_values in averaged_values_multi))

    parameter_names = [folder._parameter_name for folder in folders]
    parameter_names_str = ','.join(folder._parameter_name for folder in folders)

    # for values, averaged_values, parameter_name in zip(values_multi, averaged_values_multi, parameter_names):
    #     print(f"SHAPE: {np.array(values).shape}, {np.array(averaged_values).shape} -- {parameter_name}")

    # full graph
    plot_multi(values_multi, parameter_names, 0, N - 1, N,
               x_name='episode',
               filename=f'{parameter_names_str}_0-{N}.png',
               folder_path=folder_path)

    # averaged graph
    plot_multi(averaged_values_multi, parameter_names, 0, N_averaged - 1, N_averaged,
               x_name=f'episode batch ({plot_period} per batch)',
               filename=f'{parameter_names_str}_averaged_full.png',
               folder_path=folder_path)



def K_experiment():
    result_path = input('Path to 1 iter result folder: ')
    folders = [ResultFolder(result_path+f'_{i}') for i in range(1, 10)]
    averages_for_1 = [find_convergence(folder.prepare_phi_values()) for folder in folders]

    result_path = input('Path to 2 iter result folder: ')
    folders = [ResultFolder(result_path+f'_{i}') for i in range(1, 8)]
    averages_for_2 = [find_convergence(folder.prepare_phi_values()) for folder in folders]

    result_path = input('Path to 3 iter result folder: ')
    folders = [ResultFolder(result_path+f'_{i}') for i in range(1, 12)]
    averages_for_3 = [find_convergence(folder.prepare_phi_values()) for folder in folders]

    result_path = input('Path to 4 iter result folder: ')
    folders = [ResultFolder(result_path+f'_{i}') for i in range(1, 10)]
    averages_for_4 = [find_convergence(folder.prepare_phi_values()) for folder in folders]

    result_path = input('Path to 5 iter result folder: ')
    folders = [ResultFolder(result_path+f'_{i}') for i in range(1, 9)]
    averages_for_5 = [find_convergence(folder.prepare_phi_values()) for folder in folders]

    pixel = 1 / plt.rcParams['figure.dpi']
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=14)
    figure, ax = plt.subplots(figsize=(1200 * pixel, 1200 * pixel))
    ax.set_ylabel('Episodes for convergence')

    ax.boxplot(np.transpose(np.array(averages_for_1)), positions=[-1], widths=0.35)
    ax.boxplot(np.transpose(np.array(averages_for_2)), positions=[0], widths=0.35)
    ax.boxplot(np.transpose(np.array(averages_for_3)), positions=[1], widths=0.35)
    ax.boxplot(np.transpose(np.array(averages_for_4)), positions=[2], widths=0.35)
    ax.boxplot(np.transpose(np.array(averages_for_5)), positions=[3], widths=0.35)
    plt.ylim(0, 2700)
    plt.yticks([0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400])
    plt.xticks([-1, 0, 1, 2, 3], ['K = 1', 'K = 2', 'K = 3', 'K = 4', 'K = 5'])
    plt.savefig(os.path.join('./', 'different_k.png'), bbox_inches='tight')
    plt.close()


def multiple_model_experiment():
    result_path = input('Path to random result folder: ')
    folder = ResultFolder(result_path)
    res_random = folder.prepare_averaged_phi_values()

    result_path = input('Path to seq result folder: ')
    folder = ResultFolder(result_path)
    res_sequential = folder.prepare_averaged_phi_values()

    result_path = input('Path to empty result folder: ')
    folder = ResultFolder(result_path)
    res_empty = folder.prepare_averaged_phi_values()

    pixel = 1 / plt.rcParams['figure.dpi']
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=14)
    figure, ax = plt.subplots(figsize=(1800 * pixel, 1200 * pixel))
    ax.set_ylabel('Phi value')
    ax.set_xlabel('Episode batch (5 per episode)')

    # ax.boxplot(np.transpose(np.array(res_random)),  positions=[-1], widths=0.35)
    # ax.boxplot(np.transpose(np.array(res_sequential)),  positions=[0], widths=0.35)
    # ax.boxplot(np.transpose(np.array(res_empty)),  positions=[1], widths=0.35)
    np_linear = np.linspace(0, 500, 500)
    ax.plot(np_linear, res_random, label='random')
    ax.plot(np_linear, res_sequential, label='sequential')
    ax.plot(np_linear, res_empty, label='untrained')
    ax.legend()
    plt.ylim(0.03, 0.16)
    # plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14])
    # plt.xticks([-1, 0, 1/], ['Random training', 'Sequential Training', 'Untrained'])
    plt.savefig(os.path.join(result_path, '3models_box.png'), bbox_inches='tight')
    plt.close()


def ecmp_experiment():
    result_path = input('Path to ecmp result folder: ')
    folder = ResultFolder(result_path)
    res_ecmp = folder.prepare_phi_values()

    result_path = input('Path to ucmp result folder: ')
    folder = ResultFolder(result_path)
    res_ucmp = folder.prepare_phi_values()

    # result_path = input('Path to centralized result folder: ')
    # folder = ResultFolder(result_path)
    # res_centralized = folder.prepare_phi_values()

    result_path = input('Path to mate result folder: ')
    folder = ResultFolder(result_path)
    res_mate = folder.prepare_phi_values()

    pixel = 1 / plt.rcParams['figure.dpi']
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=14)
    figure, ax = plt.subplots(figsize=(1200 * pixel, 1200 * pixel))
    ax.set_ylabel('Phi value')
    ax.boxplot(
            np.transpose(np.array(res_ecmp)),  positions=[-1],
            widths=0.5)
    ax.boxplot(
            np.transpose(np.array(res_ucmp)),  positions=[0],
            widths=0.5)
    # ax.boxplot(
    #         np.transpose(np.array(res_centralized)),  positions=[1],
    #         widths=0.5)
    ax.boxplot(
            np.transpose(np.array(res_mate)),  positions=[1],
            widths=0.5, showfliers=False)

    plt.ylim(0.02, 0.18)
    # plt.yticks([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.15])
    plt.xticks([-1, 0, 1], ['ECMP-hashing', 'UCMP-hashing',  'MAROH'])
    plt.savefig(os.path.join(result_path, 'advantage_all.eps'), bbox_inches='tight', format='eps')
    plt.close()


def normal_plot():
    if len(sys.argv) < 2:
        print('Usage: python plot.py number [parameter_name]\n'
              'Number means plot period - the amount of episodes to aggregate into a single point on the graph\n'
              'parameter_name means which parameter to plot. Result folder has files xxx_N-NN.json files.\n'
              '  Put xxx name here (for example phi_values of tasks_placed) to use these files to produce a graph'
              '  If empty, assumes phi_values')
        exit(1)
    result_path = None
    argv = [] # argv without flags
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            if arg == '-y':
                result_path = '.'
        else:
            argv.append(arg)
    plot_period = int(argv[0])
    if len(argv) >= 2:
        parameter_names = argv[1:]
    else:
        parameter_names = ['phi_values']
    if result_path is None:
        result_path = input('Path to result folder: ')
    folders = [ResultFolder(result_path, parameter_name=parameter_name)
               for parameter_name in parameter_names]
    plot_average_and_full_multi(plot_period=plot_period, folders=folders, max_episode=None)


def statistic_boxplot():
    if len(sys.argv) < 2:
        print('Usage: python plot.py number [parameter_name]\n'
              'Number means plot period - the amount of episodes to aggregate into a single point on the graph\n'
              'parameter_name means which parameter to plot. Result folder has files xxx_N-NN.json files.\n'
              '  Put xxx name here (for example phi_values of tasks_placed) to use these files to produce a graph'
              '  If empty, assumes phi_values')
        exit(1)
    plot_period = int(sys.argv[1])
    if len(sys.argv) == 3:
        parameter_name = sys.argv[2]
    else:
        parameter_name = 'phi_values'

    paths = [
            # './newphi_fullstate_sum01dev1_phipen_83_30k_01-17-25,25-10-2023',
            # './task83_last_2023-10-26,20-48-59',
            # './test83_20k_2_2023-10-26,14-57-57',
            # './test83_20k_3_2023-10-27,02-48-56',
            # './test83_20k_4_2023-10-27,02-49-19',
            # './test83_20k_2023-10-26,14-57-19'
            './4dom_clip0.12_1541_2023-11-07,16-35-08'
    ]

    folders = [ResultFolder(result_path, parameter_name=parameter_name) for result_path in paths]
    penalty_folders = [ResultFolder(result_path, parameter_name='penalties') for result_path in paths]
    prepared_phi_values = [folder.prepare_phi_values() for folder in folders]
    prepared_penalties = [folder.prepare_phi_values() for folder in penalty_folders]
    # prepared_phi_values = [folder.prepare_averaged_phi_values() for folder in folders]

    pixel = 1 / plt.rcParams['figure.dpi']
    plt.rcParams.update({'font.size': 22})
    plt.rc('xtick', labelsize=14)
    figure, ax = plt.subplots(figsize=(2500 * pixel, 1200 * pixel))
    ax.set_ylabel(parameter_name)
    ax.set_xlabel(f'episode batch ({plot_period} per batch)')
    medianprops = dict(linestyle='-', linewidth=4, color='black')

    # xticks_array = list(range(20000 // plot_period))
    xticks_array = []
    xticks_lables = []
    for i in range((290 // plot_period)):
        all_values = []
        for phi_values, penalties in zip(prepared_phi_values, prepared_penalties):
            # all_values.extend(phi_values[plot_period * i:plot_period * (i + 1)])
            all_values.extend([phi - pen for phi, pen in zip(phi_values[plot_period * i:plot_period * (i + 1)], penalties[plot_period * i:plot_period * (i + 1)])])
            # all_values.extend(phi_values[plot_period // 5 * i:plot_period // 5 * (i + 1)])
        ax.boxplot(
                np.transpose(np.array(all_values)),  positions=[i],
                widths=0.25, showfliers=False, medianprops=medianprops)
        if i % 5 == 0:
            xticks_lables.append(str(i))
            xticks_array.append(i)

    np_phi_values = np.array([0.3428278760295368] * 290)
    np_linear = np.linspace(0, 289, 290)
    ax.plot(np_linear, np_phi_values, label='greedy algorithm')
    ax.legend()

    # plt.ylim(0.3, 0.6)
    # plt.ylim(1.9, 8.5)
    # plt.ylim(60, 84)
    plt.xticks(xticks_array, xticks_lables)
    # plt.savefig('./statistic_all.eps', bbox_inches='tight', format='eps')
    plt.savefig(f'./statistic_withgreedy_{parameter_name}_{len(paths)}_{plot_period}.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # statistic_boxplot()
    normal_plot()
    # ecmp_experiment()
    # multiple_model_experiment()

    # ----------------------------------------
    # box_values_random = []
    # for matrix_num in range(0, len(res_random) // 100):
    #     box_values_random.append(np.array(res_random[matrix_num * 100:(matrix_num + 1) * 100]))
    # np_phi_values_random = np.transpose(np.array(box_values_random))
    #
    # box_values_single = []
    # for matrix_num in range(0, len(res_single) // 100):
    #     box_values_single.append(np.array(res_single[matrix_num * 100:(matrix_num + 1) * 100]))
    # np_phi_values_single = np.transpose(np.array(box_values_single))
    #
    # box_values_sequential = []
    # for matrix_num in range(0, len(res_sequential) // 100):
    #     box_values_sequential.append(np.array(res_sequential[matrix_num * 100:(matrix_num + 1) * 100]))
    # np_phi_values_seq = np.transpose(np.array(box_values_sequential))
    #
    # box_values_empty = []
    # for matrix_num in range(0, len(res_empty) // 100):
    #     box_values_empty.append(np.array(res_empty[matrix_num * 100:(matrix_num + 1) * 100]))
    # np_phi_values_empty = np.transpose(np.array(box_values_empty))
    #
    # labels = []
    # combined_values = []
    # index = 1
    # for rv, sv, sqv, ev in zip(box_values_random, box_values_single, box_values_sequential, box_values_empty):
    #     combined_values.append(rv)
    #     combined_values.append(sv)
    #     combined_values.append(sqv)
    #     combined_values.append(ev)
    #     labels.extend((f'{index}random', f'{index}sequential', f'{index}from scratch'))
    #     index += 1
    #
    # np_combined_values = np.transpose(np.array(combined_values))
    #
    # positions_base = list(range(1, 51, 1))
    # positions1 = [x - 0.3 for x in positions_base]
    # positions2 = [x - 0.1 for x in positions_base]
    # positions3 = [x + 0.1 for x in positions_base]
    # positions4 = [x + 0.3 for x in positions_base]
    # -------------------------------------------------
    #
    # pixel = 1 / plt.rcParams['figure.dpi']
    # plt.rcParams.update({'font.size': 22})
    # plt.rc('xtick', labelsize=14)
    # figure, ax = plt.subplots(figsize=(1200 * pixel, 1200 * pixel))
    # ax.set_ylabel('Phi value')
    # # c = 'red'
    # ax.boxplot(
    #         # np_phi_values_random, positions=positions1, patch_artist=True,
    #         np.transpose(np.array(res_random)),  positions=[-1],
    #         # patch_artist=True, boxprops=dict(facecolor=c, color=c),
    #         # capprops=dict(color=c),
    #         # whiskerprops=dict(color=c),
    #         # flierprops=dict(color=c, markeredgecolor=c),
    #         # medianprops=dict(color='orange'),
    #         widths=0.35)
    # # c = 'green'
    # ax.boxplot(
    #         # np_phi_values_single, positions=positions2, patch_artist=True,
    #         np.transpose(np.array(res_sequential)),  positions=[0],
    #         # patch_artist=True, boxprops=dict(facecolor=c, color=c),
    #         # capprops=dict(color=c),
    #         # whiskerprops=dict(color=c),
    #         # flierprops=dict(color=c, markeredgecolor=c),
    #         # medianprops=dict(color='grey'),
    #         widths=0.35)
    # c = 'blue'
    # ax.boxplot(
    #         # np_phi_values_seq, positions=positions3, patch_artist=True,
    #         np.transpose(np.array(res_matedummy)), positions=[1],
    #         # patch_artist=True, boxprops=dict(facecolor=c, color=c),
    #         # capprops=dict(color=c),
    #         # whiskerprops=dict(color=c),
    #         # flierprops=dict(color=c, markeredgecolor=c),
    #         # medianprops=dict(color='cyan'),
    #         widths=0.35)
    # c = 'green'
    # ax.boxplot(
    #         # np_phi_values_empty, positions=positions4, patch_artist=True,
    #         np.transpose(np.array(res_empty)),  positions=[1],
    #         # patch_artist=True, boxprops=dict(facecolor=c, color=c),
    #         # capprops=dict(color=c),
    #         # whiskerprops=dict(color=c),
    #         # flierprops=dict(color=c, markeredgecolor=c),
    #         # medianprops=dict(color='yellow'),
    #         widths=0.35)

    # plt.xlim(0, 51)
    # plt.xticks(positions_base, positions_base)

    # np_phi_values_random = np.array(res_random)
    # np_phi_values_single = np.array(res_single)
    # np_phi_values_seq = np.array(res_sequential)
    # np_phi_values_empty = np.array(res_empty)
    # np_linear = np.linspace(0, 499, 500)
    # pixel = 1 / plt.rcParams['figure.dpi']
    # figure, ax = plt.subplots(figsize=(1200 * pixel, 800 * pixel))
    # ax.plot(np_linear, np_phi_values_random, label='random')
    # ax.plot(np_linear, np_phi_values_single, label='single')
    # ax.plot(np_linear, np_phi_values_seq, label='sequential')
    # ax.plot(np_linear, np_phi_values_empty, label='no model')
    # ax.set_xlabel('episode batch (10 per batch)')
    # ax.set_ylabel('phi value')
    # ax.legend()
        # plt.savefig(os.path.join(result_path, 'all_models.png'), bbox_inches='tight')
    # plt.close()
