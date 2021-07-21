from collections import OrderedDict
from typing import Dict
from carla import ActorSnapshot, Transform
from agents.navigation.local_planner import RoadOption
import gym
import numpy as np


class Observations(object):
    """
    Basic class that provides Observations interface. The idea is to separate concerns, i.e. each sensor
    should have a corresponding Observations class that can be then gathered and passed to Agents.

    Aside from sensors-related Observations, there are also some specific ones to provide e.g. player
    position in the world, next direction from route planner etc. Please see the basic Observations
    sublasses in gym_carla.converters.observations.
    """

    def __init__(self) -> None:
        super().__init__()

        self._bounds = OrderedDict()

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        raise NotImplementedError()

    def get_observation_space(self) -> gym.spaces.Space:
        low, high = zip(*self._bounds.values())
        return gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
