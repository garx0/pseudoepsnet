import os
import re
import json
from collections import defaultdict
from pprint import PrettyPrinter


class ResultStorage:
    def __init__(self, path):
        self._folder_path = path
        self._history = defaultdict(lambda: defaultdict(lambda: {}))
        self._parse_folder()
        self.printer = PrettyPrinter(indent=4, width=250)

    def _parse_file(self, file_obj, value_name, start):
        data_list = json.load(file_obj)
        episode_index = start
        for episode_data in data_list:
            if value_name == 'flow_split' or value_name == 'phi_values':
                # flow split and phi values have an additional value (before any actions)
                horizon_index = -1
            else:
                horizon_index = 0

            for horizon_data in episode_data:
                self._history[value_name][episode_index][horizon_index] = horizon_data
                horizon_index += 1
            episode_index += 1

    def _parse_folder(self):
        with os.scandir(self._folder_path) as files:
            for file in files:
                if not file.is_file():
                    continue
                match = re.search('([a-zA-Z0-9_-]+)_([0-9]+)-([0-9]+).json', file.name)
                if not match:
                    continue
                with open(file.path, 'r') as f:
                    self._parse_file(f, match[1], int(match[2]))
                print(f'parsed file {file.name}')

    def print_values(self, episode):
        print(f'\n\n=============== EPISODE {episode} ===============\n')
        for value_name, history_data in self._history.items():
            try:
                episode_data = history_data[episode]
            except KeyError:
                print(f'\nNo data for parameter {value_name} for episode {episode}\n')
                continue
            print(f'=============== {value_name} ===============')
            self.printer.pprint(episode_data)
            print('==============================')


if __name__ == '__main__':
    print('This tool allows to view result data from any episode.\n'
          'First enter the path to result folder, the one which contains graphs and json files.\n'
          'Then enter number of episode you want to explore\n'
          'Exit with ctrl+c.')
    result_path = input('Path to result folder: ')
    storage = ResultStorage(result_path)

    while True:
        episode = input('Number of episode: ')
        try:
            episode = int(episode)
        except ValueError:
            print(f'{episode} is not an integer value')
            continue
        storage.print_values(int(episode))

