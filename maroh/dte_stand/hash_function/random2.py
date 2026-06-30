from typing import Optional
from dte_stand.data_structures import Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction
import libscrc
import random
import copy

import logging
LOG = logging.getLogger(__name__)

def splitmix32(x):
    x = (x + 0x9e3779b9) & 0xFFFFFFFF
    x = (x ^ (x >> 16)) * 0x21f0aaad & 0xFFFFFFFF
    x = (x ^ (x >> 15)) * 0x735a2d97 & 0xFFFFFFFF
    return x ^ (x >> 15)

def splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)

class RandomHashFunction2(BaseHashFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _choose_nexthop(self, buckets: list[Bucket], flow_id: str,
                        use_flow_memory: bool = True, hash: int = None) -> Optional[GraphPathElement]:
        if buckets:
            sum_weights = sum((b.weight for b in buckets))
            new_weights = [b.weight / sum_weights for b in buckets]
            if sum_weights <= 0:
                return None
            bucket_ids = list(range(len(buckets)))
            
            hash_weight = splitmix64(hash + len(buckets)*0x9E3779B97F4A7C15) / 0xFFFFFFFFFFFFFFFF
            f = 1
            for i in range(2, len(bucket_ids)): f *= (i)
            i = len(bucket_ids) - 1
            R = hash % (f*(i+1))
            x = 0.0
            while bucket_ids:
                if i == 0:
                    return buckets[bucket_ids[0]].edge
                d = R // f
                bucket_id = bucket_ids[d]
                bucket_weight = new_weights[bucket_id]
                if bucket_weight:
                    if hash_weight < bucket_weight + x:
                        return buckets[bucket_id].edge
                R = R % f
                f = f // i
                i -= 1
                x += bucket_weight
                del bucket_ids[d]
        return None