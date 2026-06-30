from dte_stand.config import Config
from dte_stand.data_structures import InputData, HashWeights
from dte_stand.paths import DAGCalculator
# from dte_stand.hash_function import WeightedDxHashFunction
from dte_stand.hash_function.dxhash_fix import WeightedDxHashFunction
import uuid


def run_test(topo, flows, hw):
    counter_for_n2 = 0
    counter_for_n3 = 0
    for flow in flows:
        flow.flow_id = str(uuid.uuid4())
        flow_path = hash_function._flow_path(topo, flow, hw, '1', None, 1, False)
        nexthop = flow_path[0]
        if nexthop.to_ == '2':
            counter_for_n2 += 1
        else:
            counter_for_n3 += 1
    return counter_for_n2, counter_for_n3


if __name__ == '__main__':
    path = 'data_examples/rhombus_50flows'
    Config.load_config(path)
    input_data = InputData(path)
    path_calculator = DAGCalculator()
    hash_function = WeightedDxHashFunction(path_calculator)

    current_topology, _ = input_data.topology.get(0)
    current_flows = input_data.flows.get(0)
    path_calculator.prepare_iteration(current_topology)
    hash_weights = HashWeights()

    # change weights here (last parameter)
    hash_weights.put('1', '4', '3', 0, 10)
    hash_weights.put('1', '4', '2', 0, 1)

    # no point changing these for rhombus
    hash_weights.put('2', '4', '4', 0, 1)
    hash_weights.put('3', '4', '4', 0, 1)

    counter_for_n2 = 0
    counter_for_n3 = 0
    for _ in range(100):
        n2, n3 = run_test(current_topology, current_flows, hash_weights)
        counter_for_n2 += n2
        counter_for_n3 += n3
        print(f'n2: {n2} flows, n3: {n3} flows, fraction for n2: {n2 / (n2+n3)}, fraction for n3: {n3/ (n2+n3)}')

    print(f'Total from 100 runs: fraction for n2: {counter_for_n2 / (counter_for_n2 + counter_for_n3)}, '
          f'fraction for n3: {counter_for_n3 / (counter_for_n2 + counter_for_n3)}')
