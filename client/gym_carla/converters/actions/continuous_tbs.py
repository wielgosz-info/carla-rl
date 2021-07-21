from typing import OrderedDict
from gym_carla.converters.actions.actions_converter import ActionsConverter
import gym
import numpy as np
from carla import VehicleControl


class ContinuousTBSActionsConverter(ActionsConverter):
    """
    Converter for continuous Throttle, Brake, Steer actions
    """

    def __init__(self):
        super().__init__()
        self._bounds = OrderedDict(
            throttle=[0.0, 1.0],
            brake=[0.0, 1.0],
            steer=[-1.0, 1.0]
        )

    def action_to_control(self, action, ego_vehicle_snapshot=None):
        control = VehicleControl()

        for idx, (key, bounds) in enumerate(self._bounds.items()):
            setattr(control, key, min(bounds[1], max(bounds[0], action[idx])))

        return control
