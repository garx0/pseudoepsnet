import os
import networkx
import matplotlib.pyplot as plt
import numpy as np
import json
import multiprocessing as mp
from logging import getLogger
from dte_stand.config import Config
from dte_stand.history import HistoryTracker
from time import sleep

LOG = getLogger(__name__)

N_IO_TRIES = 12

class PhiCalculator:
    _all_phi_values: list[float] = list() # episodes values of all iterations
    _episodes_phi_values: list[tuple[float, list[float]]] = list() # episodes values of current iteration
    _horizons_phi_values: list[float] = list() # horizons values of current episode
    _iteration_horizons: list[float] = list() # horizon values of current iteration
    _all_horizons_phi_values: list[float] = list() # horizon values of the whole experiment
    _eval_values: list[float] = list()
    _eval_points: list[int] = list()
    _eval_values_full: list[list[int]] = list()
    _eval_msg_done = []
    _eval_msg_possible = []
    _eval_inputs: list[str] = list()
    _plot_number = 0
    _plot_folder = './'
    _tracker = None

    _message_iterations_done = []
    _message_iterations_possible = []

    @classmethod
    def init_tracker(cls):
        cls._tracker = HistoryTracker()

    @classmethod
    def set_plot_folder(cls, path: str) -> None:
        cls._plot_folder = path

    @classmethod
    def end_episode(cls) -> None:
        if cls._tracker:
            cls._tracker.add_value('phi_values', cls._horizons_phi_values)
            cls._tracker.end_iteration()
        try:
            episode_phi_value = cls._horizons_phi_values.pop()
        except IndexError:
            return
        cls._episodes_phi_values.append((episode_phi_value, cls._horizons_phi_values))
        cls._all_horizons_phi_values.extend(cls._horizons_phi_values)
        cls._iteration_horizons.extend(cls._horizons_phi_values)
        cls._horizons_phi_values.clear()

    @classmethod
    def end_iteration_and_plot_graph(cls) -> None:
        phi_values = (phi for phi, _ in cls._episodes_phi_values)
        cls._all_phi_values.extend(phi_values)
        #cls.plot_full(all_iterations=False)
        # cls.plot_iteration_horizons()
        cls._episodes_phi_values.clear()
        cls._horizons_phi_values.clear()
        # cls._iteration_horizons.clear()
        cls._plot_number = 0

    @classmethod
    def plot_iteration_horizons(cls) -> None:
        cls._plot(cls._iteration_horizons, 0, len(cls._iteration_horizons)-1, len(cls._iteration_horizons), 'horizon', 'horizons.png')
        return

    @classmethod
    def calculate_phi(cls, topology: networkx.MultiDiGraph, eval = False) -> float:
        # calculate phi as it is in problem statement
        number_of_edges = topology.number_of_edges()
        average_load: float = 0.0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            average_load += float(edge_data['current_bandwidth']) / edge_data['bandwidth']
        average_load /= number_of_edges

        deviation: float = 0.0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            deviation += pow(float(edge_data['current_bandwidth']) / edge_data['bandwidth'] - average_load, 2)
        phi = deviation / number_of_edges
        if eval == False:
            cls._horizons_phi_values.append(phi)
        return phi

    @classmethod
    def calculate_max_value(cls, topology: networkx.MultiDiGraph) -> float:
        # calculate phi but as just maximum link bw instead
        maximum: int = 0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            value = edge_data['current_bandwidth']
            if value > maximum:
                maximum = value
        cls._horizons_phi_values.append(maximum)
        return maximum

    @classmethod
    def calculate_max_util(cls, topology: networkx.MultiDiGraph, eval = False) -> float:
        # calculate phi but as just maximum link utilization instead
        maximum: float = 0.0
        for edge in topology.edges(data=True):
            _, _, edge_data = edge
            value = edge_data['current_bandwidth'] / edge_data['bandwidth']
            if value > maximum:
                maximum = value
        if eval == False:
            cls._horizons_phi_values.append(maximum)
        return maximum

    @classmethod
    def _plot(cls, values: list, start: int, end: int, amount: int, x_name: str, filename: str, eval = None, x_values: list = None):
        """ Running in different process to avoid matplotlib memory leaks """
        p = mp.Process(target=cls._plot_impl, args=(cls._plot_folder, values, start, end, amount, x_name, filename, eval, x_values))
        p.start()
        p.join()

    @classmethod
    def _plot_impl(cls, plot_folder: str, values: list, start: int, end: int, amount: int, x_name: str, filename: str, eval = None, x_values: list = None):
        np_phi_values = np.array(values)
        np_linear = np.linspace(start, end, amount) if x_values is None else x_values
        pixel = 1/plt.rcParams['figure.dpi']
        figure, ax = plt.subplots(figsize=(1200*pixel, 800*pixel))
        ax.plot(np_linear, np_phi_values, label='train' if amount > 0 else None)
        if eval:
            eval_pts = np.array(eval[0])
            eval_values = np.array(eval[1])
            ax.plot(eval_pts, eval_values, label='eval')
        ax.set_xlabel(x_name)
        ax.set_ylabel('phi value')
        ax.legend()
        ok = False
        for _ in range(N_IO_TRIES):
            try:
                plt.savefig(os.path.join(plot_folder, filename))
                ok = True
                break
            except Exception as e:
                print(f"IOERROR: {e}")
                sleep(5)
        if not ok:
            print(f"wasn't able to plot due to repeating IOERROR, ignored")
            pass
        plt.close('all')


    @classmethod
    def plot_experiment(cls) -> None:
        """
        plot all horizons of all iterations
        """
        cls._plot(cls._all_horizons_phi_values, 0, len(cls._all_horizons_phi_values) - 1, len(cls._all_horizons_phi_values), 'episode', f'plot_horizons_full.png', None)
        return

    @classmethod
    def plot_full(cls, all_iterations: bool) -> None:
        """
        Plot two graphs:
          1. full graph for all episodes or iterations
          2. averaged graph for all episodes or iterations (one point for every plot period)

        :param all_iterations: if true, plot graph for all iterations. Else, plot only for current iteration
        :return: None
        """
        config = Config.config()
        plot_period = config.plot_period

        # if all_iterations:
        #     phi_values = cls._all_phi_values
        # else:
        #     phi_values = [phi for phi, _ in cls._episodes_phi_values]

        phi_values = [phi for phi, _ in cls._episodes_phi_values]

        if not phi_values:
            return

        averaged_values = []
        avg_values_x = []
        number_of_points = (len(phi_values) // plot_period)
        if len(phi_values) % plot_period != 0:
            number_of_points += 1

        for i in range(number_of_points):
            start = i * plot_period
            end = min((i + 1) * plot_period, len(phi_values))
            averaged_values.append(sum(phi_values[start:end]) / (end - start))
            avg_values_x.append(end)


        if len(cls._eval_values) > 0:
            eval = (cls._eval_points, cls._eval_values)
            eval_full = (cls._eval_points, cls._eval_values_full,
                         cls._eval_msg_done, cls._eval_msg_possible, cls._eval_inputs)

            ok = False
            for _ in range(N_IO_TRIES):
                try:
                    with open(os.path.join(cls._plot_folder, f'eval.json'), 'w') as f:
                        json.dump(eval, f)
                    ok = True
                    break
                except Exception as e:
                    print(f"IOERROR: {e}")
                    sleep(5)
            if not ok:
                raise Exception("couldn't write eval.json due to repeated IOERROR")

            ok = False
            for _ in range(N_IO_TRIES):
                try:
                    with open(os.path.join(cls._plot_folder, f'eval_full.json'), 'w') as f:
                        json.dump(eval_full, f)
                    ok = True
                    break
                except Exception as e:
                    print(f"IOERROR: {e}")
                    sleep(5)
            if not ok:
                raise Exception("couldn't read eval_full.json due to repeated IOERROR")

            cls._plot([], 0, 0, 0, 'episode', f'eval.png', eval, x_values=[])
        else:
            eval = None

        # full graph
        cls._plot(phi_values, 0, len(phi_values) - 1, len(phi_values), 'episode', f'plot_0-{len(phi_values)}.png', eval)

        # averaged graph
        cls._plot(averaged_values, 0, len(averaged_values) - 1, len(averaged_values), 'episode batch',
                  'plot_averaged.png' if not all_iterations else 'plot_averaged_full.png', eval, avg_values_x)

    @classmethod
    def plot_result(cls) -> None:
        """
        Plot graph for one plot period

        :return: None
        """
        config = Config.config()
        plot_period = config.plot_period
        cls._plot_number += 1
        episode_phi_values = [phi for phi, _ in cls._episodes_phi_values]

        start = (cls._plot_number - 1) * plot_period
        end = min(cls._plot_number * plot_period, len(episode_phi_values))

        # nonzero_phis = [p for p in episode_phi_values[start:end] if p != 0]
        # LOG.info(f'start = {start} end = {end} zero = {end - start - len(nonzero_phis)} nonzero = {len(nonzero_phis)} zero_fraction = {1 - len(nonzero_phis)/(end - start)}')
        cls._plot(episode_phi_values[start:end], start, end - 1, end - start, 'episode', f'plot_{start}-{end - 1}.png')

        cls._plot(cls._iteration_horizons, 0, len(cls._iteration_horizons)-1, len(cls._iteration_horizons), 'horizon', f'horizons_{start}-{end - 1}.png')
        cls._iteration_horizons.clear()

        ok = False
        for _ in range(N_IO_TRIES):
            try:
                with open(os.path.join(cls._plot_folder, f'messages_{start}-{end - 1}.json'), 'w') as fp:
                    json.dump({
                        "message_iterations_done": cls._message_iterations_done,
                        "message_iterations_possible": cls._message_iterations_possible
                    }, fp)
                ok = True
                break
            except Exception as e:
                print(f"IOERROR: {e}")
                sleep(5)
        if not ok:
            raise Exception("couldn't read message iterations values due to repeated IOERROR")

        cls._message_iterations_done = []
        cls._message_iterations_possible = []

    @classmethod
    def get_average(cls):
        sum = 0
        for value in cls._all_phi_values:
            sum += value
        avg = float(sum) / len(cls._all_phi_values)
        cls._all_phi_values.clear()
        cls._horizons_phi_values.clear()
        cls._episodes_phi_values.clear()
        return avg
