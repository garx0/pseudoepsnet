# import json
# import uuid
# from pydantic import BaseModel, validator, PositiveInt, Field

import json
import uuid
from pydantic import BaseModel, RootModel, field_validator, PositiveInt, Field, ConfigDict
from typing import Dict, Any, Optional
from collections.abc import Iterator
from time import sleep

import logging
LOG = logging.getLogger(__name__)

N_IO_TRIES = 12

def struuid():
    return str(uuid.uuid4())


class Flow(BaseModel):
    start: str
    end: str
    all_bandwidth: dict[str, PositiveInt]
    start_time: int
    end_time: int
    bandwidth: Optional[int] = None
    flow_id: str = Field(default_factory=struuid)
    hash: int = None

    @field_validator('end')
    @classmethod
    def start_differs_from_end(cls, v: str, info: Any) -> str:
        if v == info.data['start']:
            raise ValueError('Flow start and end points are same')
        return v

    @field_validator('end_time')
    @classmethod
    def start_before_end(cls, v: int, info: Any) -> int:
        if v <= info.data['start_time']:
            raise ValueError('Flow start and end times are incorrect')
        return v

    model_config = ConfigDict(
        exclude={"flow_id", "bandwidth"}
    )


class InputFlows(RootModel):
    root: list[Flow]

    def __iter__(self) -> Iterator[Flow]:
        return iter(self.root)

    def append(self, item: Flow):
        self.root.append(item)

    def __getitem__(self, item):
        return self.root[item]


class Flows:
    def __init__(self, path_to_flows):
        data = None
        for _ in range(N_IO_TRIES):
            try:
                with open(path_to_flows, 'r') as f:
                    data = json.load(f)
                break
            except Exception as e:
                print(f"IOERROR: {e}")
                sleep(5)
        if data is None:
            raise Exception(f"wasn't able to read {path_to_flows}")
        self._flows: InputFlows = InputFlows.model_validate(data["flows"])

    def get(self, current_time: int) -> list[Flow]:
        try:
            needed_flows = (f for f in self._flows if f.start_time <= current_time < f.end_time)
            result_flows = []
            for flow in needed_flows:
                latest_change = [t for t in flow.all_bandwidth if int(t) <= current_time][-1]
                flow.bandwidth = flow.all_bandwidth[latest_change]
                result_flows.append(flow)
            return result_flows
        except KeyError:
            return []
