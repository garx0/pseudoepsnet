from typing import Optional
from dte_stand.data_structures import Bucket, GraphPathElement
from dte_stand.hash_function.base import BaseHashFunction

import logging
LOG = logging.getLogger(__name__)


class DummyHashFunction(BaseHashFunction):
    def _choose_nexthop(self, buckets: list[Bucket], flow_id: str,
                        use_flow_memory: bool = True) -> Optional[GraphPathElement]:
        try:
            return buckets[0].edge
        except IndexError:
            # empty list of buckets
            return None
