from typing import Optional
from dte_stand.data_structures import Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction
import libscrc
import random
import copy

import logging
LOG = logging.getLogger(__name__)


class WeightedDxHashFunction(BaseHashFunction):
    def __init__(self, *args, **kwargs):
        self._rng = random.Random()
        self._flow_to_bucket_map: dict[tuple[str, str], GraphPathElement] = {}
        super().__init__(*args, **kwargs)

    def _choose_nexthop(self, buckets: list[Bucket], flow_id: str,
                        use_flow_memory: bool = True, hash: int = None) -> Optional[GraphPathElement]:
        if buckets:
            self._rng.seed(flow_id)
            if use_flow_memory and (self._flow_to_bucket_map.get((flow_id, buckets[0].edge.from_))
                                    and self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)] in
                                            list(bucket.edge for bucket in buckets)):
                return self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)]
            max_weight = max((b.weight for b in buckets))
            if max_weight <= 0:
                return None
            bucket_ids = list(range(len(buckets)))
            while bucket_ids:
                key = self._rng.randint(0, 10000)
                bucket_number = key % len(bucket_ids)
                bucket_weight = buckets[bucket_ids[bucket_number]].weight / max_weight
                if bucket_weight:
                    hash_value = libscrc.crc32(bytes(key))
                    hash_weight = float(hash_value & 0xffffffff) / 2 ** 32
                    if hash_weight < bucket_weight:
                        if use_flow_memory:
                            self._flow_to_bucket_map[(flow_id, buckets[0].edge.from_)] = \
                                buckets[bucket_ids[bucket_number]].edge
                        return buckets[bucket_ids[bucket_number]].edge
                del bucket_ids[bucket_number]
        return None