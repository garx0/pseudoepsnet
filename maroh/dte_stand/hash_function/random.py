from typing import Optional
from dte_stand.data_structures import Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction
import libscrc
import random
import copy

import logging
LOG = logging.getLogger(__name__)


class RandomHashFunction(BaseHashFunction):
    def __init__(self, *args, **kwargs):
        self._rng = random.Random()
        super().__init__(*args, **kwargs)

    def _choose_nexthop(self, buckets: list[Bucket], flow_id: str,
                        use_flow_memory: bool = True, hash: int = None) -> Optional[GraphPathElement]:
        if buckets:
            self._rng.seed(flow_id)
            sum_weights = sum((b.weight for b in buckets))
            new_weights = [b.weight / sum_weights for b in buckets]
            if sum_weights <= 0:
                return None
            bucket_ids = list(range(len(buckets)))
            self._rng.shuffle(bucket_ids)
            x = 0.0
            hash_weight = self._rng.random()
            for bucket_id in bucket_ids:
                bucket_weight = new_weights[bucket_id]
                if bucket_weight:
                    
                    if hash_weight < bucket_weight + x:
                        return buckets[bucket_id].edge
                x += bucket_weight
            return buckets[bucket_ids[len(bucket_ids)-1]].edge
        return None