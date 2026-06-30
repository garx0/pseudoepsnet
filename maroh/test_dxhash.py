from dte_stand.data_structures.paths import GraphPathElement
from dte_stand.data_structures.hash_weights import Bucket
from dte_stand.hash_function.dxhash import WeightedDxHashFunction
import uuid

import unittest

edges = [
    [("A", "B", 0)],
    [("A", "B", 1)],
    [("A", "B", 0), ("AA", "BB", 1), ("AAA", "BBB", 2)],
    [("Warsaw", "Oslo", 0), ("Amsterdam", "Monaco", 1)],
    [("Dublin", "Budapest", 0), ("Madrid", "Oslo", 1), ("Lisbon", "Copenhagen", 2),
    ("Helsinki", "Kiev", 3), ("Stockholm", "Bern", 4), ("San-Marino", "Monaco", 5),
    ("Riga", "Prague", 6), ("Riga", "Zagreb", 7), ("Valletta", "San-Marino", 8),
    ("Vilnius", "Skopje", 9), ("Luxembourg", "Chisinau", 10), ("Amsterdam", "Monaco", 11)],
    [('Berlin', "Bratislava", 0), ("Moscow", "Stockholm", 1), ("Stockholm", "Bern", 2),
     ("Helsinki", "Kiev", 3), ("Amsterdam", "Luxembourg", 4), ("Copenhagen", "Moscow", 5),
     ("Amsterdam", "Moscow", 6), ("Amsterdam", "Vatican", 7), ("Amsterdam", "Valletta", 8),
     ("Valletta", "Riga", 9), ("Ljubljana", "Rome", 10), ("Moscow", "Monaco", 11),
     ("Zagreb", "Dublin", 12), ("Chisinau", "Vatican", 13), ("Budapest", "Bern", 14)],
    [('Berlin', "Bratislava", 0), ("Moscow", "Stockholm", 1), ("Stockholm", "Bern", 2),
     ("Helsinki", "Kiev", 3), ("Amsterdam", "Monaco", 4), ("Copenhagen", "Moscow", 5),
     ("Helsinki", "Monaco", 6), ("Warsaw", "Chisinau", 7), ("Dublin", "Monaco", 8),
     ("Stockholm", "Monaco", 9), ("Warsaw", "Stockholm", 10), ("Stockholm", "Moscow", 11),
     ("Amsterdam", "Vilnius", 12), ("Riga", "Copenhagen", 13), ("Amsterdam", "Ljubljana", 14),
     ("Warsaw", "London", 15), ("Valletta", "Budapest", 16), ("Amsterdam", "Vatican", 17),
     ("Amsterdam", "Dublin", 18), ("Riga", "San-Marino", 19)],
]

weights = [
    [0],
    [1],
    [0, 0, 0],
    [0, 223],
    [ 0, 0, 0, 0, -1, 0, 0, 1, 0, -1, 0, 0],
    [-12, -15554355, -166, -19122, 0, -122, 0, -3889, 0, 9, 0, -188459309, -1, -33, 12399087],
    [ 0, -12277, 0, -94893484988888894, -1, 0, -62737636273622, 0, -123, 0, -18129192334,
    -9223372036854775, 0, 9223372036854775878430420949, -1, -92233720368547, -6, -3, -942839288999885311, 1]
]


class TestWeightedDxHash(unittest.TestCase):
    def get_edge(self, path_elem: GraphPathElement):
        if path_elem != None:
            return (path_elem.from_, path_elem.to_, path_elem.index)
        return path_elem

    def fill_buckets(self, test_number):
        self.hash_weights = []
        self.flow_id = str(uuid.uuid4())
        for i in range(len(edges[test_number])):
            nexthop = GraphPathElement(edges[test_number][i][0], edges[test_number][i][1], edges[test_number][i][2])
            obj = Bucket(nexthop, weights[test_number][i])
            self.hash_weights.append(obj)
        self.hash = WeightedDxHashFunction(self.hash_weights)

    def test_no_nodes(self):
        self.fill_buckets(0)
        nexthop = self.hash._choose_nexthop(self.hash_weights, self.flow_id)
        self.assertEqual(self.get_edge(nexthop), None)

    def test_single_node(self):
        self.fill_buckets(1)
        nexthop = self.hash._choose_nexthop(self.hash_weights, self.flow_id)
        self.assertEqual(self.get_edge(nexthop), ("A", "B", 1))

    def test_all_failed_nodes(self):
        self.fill_buckets(2)
        nexthop = self.hash._choose_nexthop(self.hash_weights, self.flow_id)
        self.assertEqual(self.get_edge(nexthop), None)

    def test_single_working_node(self):
        self.fill_buckets(3)
        nexthop = self.hash._choose_nexthop(self.hash_weights, self.flow_id)
        self.assertEqual(self.get_edge(nexthop), ("Amsterdam", "Monaco", 1))

    def test_binary_weights(self):
        self.fill_buckets(4)
        nexthop = self.hash._choose_nexthop(self.hash_weights, self.flow_id)
        self.assertEqual(self.get_edge(nexthop), ("Riga", "Zagreb", 7))

    def test_negative_weights(self):
        self.fill_buckets(5)
        nexthop = self.hash._choose_nexthop(self.hash_weights, self.flow_id)
        self.assertEqual(self.get_edge(nexthop), ("Budapest", "Bern", 14))

    def test_combined_case(self):
        self.fill_buckets(6)
        nexthop = self.hash._choose_nexthop(self.hash_weights, self.flow_id)
        self.assertEqual(self.get_edge(nexthop), ("Riga", "Copenhagen", 13))


if __name__ == "__main__":
    unittest.main()
