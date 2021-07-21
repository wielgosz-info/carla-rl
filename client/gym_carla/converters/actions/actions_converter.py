from collections import OrderedDict
import gym
import numpy as np
from carla import VehicleControl, ActorSnapshot


class ActionsConverter(object):
    def __init__(self):
        super().__init__()
        self._bounds = OrderedDict()

    def get_action_space(self) -> gym.spaces.Space:
        low, high = zip(*self._bounds.values())
        return gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def action_to_control(self, action, ego_vehicle_snapshot: ActorSnapshot = None) -> VehicleControl:
        raise NotImplementedError()
