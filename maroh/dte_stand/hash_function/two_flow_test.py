from typing import Optional
from dte_stand.data_structures import Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction

import logging
LOG = logging.getLogger(__name__)


class TwoFlowTestHash(BaseHashFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _choose_nexthop(self, buckets: list[Bucket], flow_id: str,
                        use_flow_memory: bool = True) -> Optional[GraphPathElement]:
        if len(buckets) != 2:
            return buckets[0].edge

        if abs(buckets[0].weight - buckets[1].weight) > 0:
            return buckets[0].edge
        else:
            if flow_id == '30a71755-544e-44ca-a358-519b9f8a6db7':
                return buckets[0].edge
            else:
                return buckets[1].edge
