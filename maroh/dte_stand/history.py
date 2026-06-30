import os
import json
import copy
import numpy as np
from numpy import ndarray, float32
from collections import defaultdict
from dte_stand.config import Config
from time import sleep

N_IO_TRIES = 12

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


class HistoryTracker:
    _result_folder = './'

    @classmethod
    def set_result_folder(cls, folder):
        cls._result_folder = folder

    def __init__(self):
        config = Config.config()
        self._plot_period = config.plot_period
        self._iteration = 0
        self._history: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._plot_number = 0

    def add_value(self, tracked_name, value):
        if tracked_name == "phi_values":
            self._history[tracked_name][tracked_name].append(copy.deepcopy(value))
        else:
            for subparam_name, values in value.items():
                self._history[tracked_name][subparam_name].append(copy.deepcopy(values))

    def _save_to_file(self):
        self._plot_number += 1
        for name, values in self._history.items():
            if name == "phi_values":
                values_json = json.dumps(values[name], cls=TensorEncoder)
                # with open(os.path.join(
                #             HistoryTracker._result_folder, f'{name}_'
                #                                            f'{(self._plot_number - 1) * self._plot_period}-'
                #                                            f'{self._plot_number * self._plot_period - 1}.json'), 'w') as f:
                #     f.write(values_json)
                ok = False
                for _ in range(N_IO_TRIES):
                    try:
                        with open(os.path.join(
                                HistoryTracker._result_folder, f'{name}_'
                                                            f'{(self._plot_number - 1) * self._plot_period}-'
                                                            f'{self._plot_number * self._plot_period - 1}.json'), 'w') as f:
                            f.write(values_json)
                        ok = True
                        break
                    except Exception as e:
                        print(f"IOERROR: {e}")
                        sleep(5)
                if not ok:
                    raise Exception("couldn't write values due to repeated IOERROR")
            else:
                values_fmt = dict((key, np.array(val)) for key, val in values.items())
                path = os.path.join(HistoryTracker._result_folder, f'{name}_'
                                    f'{(self._plot_number - 1) * self._plot_period}-'
                                    f'{self._plot_number * self._plot_period - 1}.npz')
                np.savez(path, **values_fmt)

    def end_iteration(self):
        self._iteration += 1
        if self._iteration % self._plot_period == 0:
            self._save_to_file()
            self._history.clear()

    def reset(self):
        self._iteration = 0
        self._plot_number = 0
        self._history.clear()
