import yaml
import os
from typing import Optional
from pydantic import BaseModel, PositiveInt
from time import sleep

N_IO_TRIES = 12

class YamlConfig(BaseModel):
    hash_function: str
    algorithm: str
    path_calculator: str
    phi: str
    iterations: PositiveInt
    plot_period: int
    lsdb_period: PositiveInt
    retain_weights: bool
    log_path: str
    log_level: str
    debug_check_cycles: int
    store_hashweights: int
    store_nexthops: int
    split_flows: bool = False
    alg_cfg: dict = None # algorithm config, parsed later by the algorithm (in case of MAROH - parsed into a MateConfig object in run_experiment)


class Config:
    _config: Optional[YamlConfig] = None

    @classmethod
    def load_config(cls, path_to_folder: str, modifier: str = '') -> None:
        # with open(os.path.join(path_to_folder, f'config{modifier}.yaml'), 'r') as f:
        #     config_dict = yaml.load(f, Loader=yaml.Loader)

        config_dict = None
        for _ in range(N_IO_TRIES):
            try:
                with open(os.path.join(path_to_folder, f'config{modifier}.yaml'), 'r') as f:
                    config_dict = yaml.load(f, Loader=yaml.Loader)
                break
            except Exception as e:
                print(f"IOERROR: {e}")
                sleep(5)
        if config_dict is None:
            raise Exception("couldn't read config due to repeated IOERROR")

        #alg_cfg = config_dict.pop('alg_cfg')
        cls._config = YamlConfig.parse_obj(config_dict)
        #cls._config

    @classmethod
    def config(cls) -> Optional[YamlConfig]:
        return cls._config
