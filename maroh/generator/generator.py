from dte_stand.data_structures.flows import InputFlows, Flow
from collections import defaultdict
import random
import itertools


class UniformFlowGenerator:
    """
    Given the traffic matrix, generate flows, their bandwidths and duration using uniform distribution
    """
    def __init__(self, max_flow_amount: int, min_flow_duration: int, max_flow_duration: int):
        """
        :param max_flow_amount: limits the maximum amount of flows that can exist between any pair of nodes at any time
            actual amount of flows is a random number between this and the amount that carried from previous step
            of the generation. So setting this number higher means there will be more flows generated but with
            smaller bandwidths
        :param min_flow_duration: all generated flows will have at least this duration
        :param max_flow_duration: all generated flows will have at most this duration
        """
        self.max_flow_amount = max_flow_amount
        self.min_flow_duration = min_flow_duration
        self.max_flow_duration = max_flow_duration
        random.seed()

    def _expire_flows(self, current_flows: dict[str, dict[str, list[Flow]]],
                      all_flows: InputFlows, current_time: int) -> dict[str, dict[str, list[Flow]]]:
        """
        Move expired flows from the list of flows currently being processed (current_flows)
            to the list of all generated flows (all_flows).
        """
        new_flow_dict: dict[str, dict[str, list[Flow]]] = defaultdict(lambda: defaultdict(lambda: []))
        for source, dst_dict in current_flows.items():
            for destination, flow_list in dst_dict.items():
                for flow in flow_list:
                    if int(flow.end_time) >= current_time:
                        new_flow_dict[source][destination].append(flow)
                    else:
                        all_flows.append(flow)
        return new_flow_dict

    def _uniform_sampling(self, total: int, amount: int) -> list[int]:
        """
        uniformly generate integers that sum to a given value
            it is done by generating amount - 1 integers, then splitting the [0;total] segment into 'amount' segments
            and taking the length of these segments as generated numbers
        :param total: given sum
        :param amount: amount of integers to generate
        :return: list of generated numbers
        """
        if total == 0:
            return []
        if amount == 1 or total < amount:
            return [total]
        generated: list[int] = random.sample(range(1, total), amount - 1)
        generated.sort()
        return [generated[0]] + [generated[i+1] - generated[i] for i in range(amount - 2)] + [total - generated[-1]]

    def generate(self, traffic_matrix_list: list[dict[str, dict[str, int]]], period: int) -> InputFlows:
        """
        Main function to generate flows
        :param traffic_matrix_list: traffic matrix is a list of measurement points.
            Each measurement point gives the amount of traffic for each pair of (source, destination) graph nodes.
            So each element of the list must be a dict with key: source node, value: dst_dict
            Each dst_dict must be a dict with key: destination node, value: amount of traffic
            Graph nodes are just names.
        :param period: time in milliseconds between measurement points
        :return: InputFlows data structure object that holds all generated flows
        """
        all_flows = InputFlows.parse_obj([])
        current_flows: dict[str, dict[str, list[Flow]]] = defaultdict(lambda: defaultdict(lambda: []))
        current_time = 0
        for traffic_matrix in traffic_matrix_list:
            current_flows = self._expire_flows(current_flows, all_flows, current_time)

            for source, dst_list in traffic_matrix.items():
                for destination, traffic_value in dst_list.items():
                    if traffic_value == 0:
                        continue
                    # decide how many flow bandwidths need to be generated
                    flow_list = current_flows[source][destination]
                    current_flow_amount = len(flow_list)
                    if current_flow_amount >= self.max_flow_amount:
                        flow_amount = current_flow_amount
                    else:
                        flow_amount = int(random.uniform(current_flow_amount, self.max_flow_amount - 1)) + 1

                    # generate bandwidth for every flow
                    flow_bandwidths = self._uniform_sampling(traffic_value, flow_amount)

                    # set new bandwidth for every flow and create new flows
                    for bandwidth, flow in itertools.zip_longest(flow_bandwidths, flow_list):
                        assert bandwidth
                        if flow:
                            flow.all_bandwidth[str(current_time)] = bandwidth
                        else:
                            duration = int(random.uniform(self.min_flow_duration, self.max_flow_duration))
                            all_bandwidth = {str(current_time): bandwidth}
                            new_flow = Flow(
                                    start=source, end=destination, start_time=current_time,
                                    end_time=current_time + duration, all_bandwidth=all_bandwidth
                            )
                            current_flows[source][destination].append(new_flow)

            current_time += period

        # expire all flows
        self._expire_flows(current_flows, all_flows, current_time + self.max_flow_duration + 1)
        return all_flows
