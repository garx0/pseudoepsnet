import yaml
import os
from typing import Optional
from pydantic import BaseModel, PositiveInt

class SaActorCfg(BaseModel):
    mpnn_mode: str = '1'
    memory_lookup: str = 'state' # state, state_neigh
    networks: str = '1313' # CreateMessage and LinkUpdate networks for local aggregation at steps 1 and 3

    update_memory: bool = True

    use_memory: bool = False
    memory_type: str = 'cluster' # cluster, gpr, knn, bypass, epsilon_net
    # number of saved states
    memory_size: int = 256
    threshold: float = 0.001
    P: int = 3
    var: float = 0.01
    first_iter_active: int = 0
    representatives_path: str = 'representatives'
    all_agents: bool = False

    use_gat: bool = False
    # number of attention heads
    gat_num_heads: int = 20
    states_logging: bool = False

class ActorCfg(BaseModel):
    use_memory: bool = False
    # number of saved states
    memory_size: int = 512

    use_gat: bool = False
    # number of attention heads
    gat_num_heads: int = 2


class ActionDescription(BaseModel):
    action: bool = False
    value: float = 0.


class MateActions(BaseModel):
    addition: ActionDescription
    subtraction: ActionDescription
    multiplication: ActionDescription
    multiplication2: ActionDescription = ActionDescription()
    division: ActionDescription
    division2: ActionDescription = ActionDescription()
    zero: ActionDescription


class MateConfig(BaseModel):
    horizons: int
    episodes: int
    eval_period: int = -1
    init_weights: str = 'equal' # equal, random, beta
    change_sample_period: int = -1
    gamma: float
    gae_lambda: float
    grad_clip: float = 1.0
    entropy_coef: float = 0.001
    clip_state: bool = False
    normalize_rewards: bool = True
    reward: str
    reward_computation: str
    min_weight: float
    max_weight: float
    greedy_epsilon: float
    actions: MateActions
    message_iterations: int = -1
    n_without_update: int = 1
    batch_size: int = 64
    lr_actor: float = 0.0003
    lr_critic: float = 0.0003
    random_sample: bool = True # shuffle flows in training dataset. set to False for collecting states and greedy actions data
    # actor-specific configuration, is parsed by actor itself
    actor_cfg: Optional[dict]
