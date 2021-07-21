from typing import OrderedDict
from gym_carla.converters.actions.actions_converter import ActionsConverter
import gym
import numpy as np
from carla import VehicleControl


class ContinuousTBSActionsConverter(ActionsConverter):
    '''
    Converter for continuous Throttle, Brake, Steer actions
    '''

    def __init__(self):
        self.__limits = OrderedDict(
            throttle=[0.0, 1.0],
            brake=[0.0, 1.0],
            steer=[-1.0, 1.0]
        )

    def get_action_space(self):
        low, high = zip(*self.__limits.values())
        return gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def action_to_control(self, action, last_ego_vehicle_snapshot=None):
        control = VehicleControl()

        for idx, (key, limits) in enumerate(self.__limits.items()):
            setattr(control, key, min(limits[1], max(limits[0], action[idx])))

        return control
