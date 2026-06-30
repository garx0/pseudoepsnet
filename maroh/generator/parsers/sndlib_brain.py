import glob
import os
import re
from collections import defaultdict


def parse_matrix(file_content: str) -> list[tuple[str, str, int]]:
    return re.findall(
        '[a-zA-Z0-9]*_[a-zA-Z0-9]*'  # name of a single demand
        '[ \t]*\\([ \t]*'  # whitespaces and left parenthesis
        '([a-zA-Z0-9]*) ([a-zA-Z0-9]*)'  # node names (needed info)
        '[ \t]*\\)[ \t]*[0-9]*[ \t]*'  # whitespaces and right parenthesis
        '([0-9]*).[0-9]*',  # demand value (needed info but fractions are discarded)
        file_content, flags=re.MULTILINE
    )


def parse_all(path_to_folder: str) -> list[dict[str, dict[str, int]]]:
    # get file list of files that may be sndlib's demand matrices
    file_list = glob.glob(os.path.join(path_to_folder, '*.txt'))
    if not file_list:
        raise Exception(f'No files of needed format found in {path_to_folder}')

    # alphabetical sorting seems to be enough for this dataset
    file_list.sort()

    # parse every file
    traffic_matrices: list[dict[str, dict[str, int]]] = []
    for file in file_list:
        with open(file, 'r') as f:
            file_content = f.read()
            demand_data = parse_matrix(file_content)
            if not demand_data:
                print(f'{file} is not an sndlib dataset file')
                continue

        # construct dict and save it
        traffic_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        for source, destination, bandwidth in demand_data:
            # this dataset's topology is too detailed
            # there are a lot of small nodes (labelled as LLLDD, where D is a digit, L is a letter)
            # and matrix gives demands between these small nodes (like, AAA11 -> AAA12)
            # due to how topology os constructed, there is no point for our algorithm to consider these demands
            # so we need to unify all small nodes into a single large node (AAA11 + AAA12 -> just AAA)
            # and matrix will only contain demands between large nodes

            # remove numbers from name
            source_name = re.sub('[0-9]+', '', source)
            destination_name = re.sub('[0-9]+', '', destination)
            if source_name == destination_name:
                # skip same names
                continue
            traffic_matrix[source_name][destination_name] += int(bandwidth)
        traffic_matrices.append(traffic_matrix)

    return traffic_matrices
