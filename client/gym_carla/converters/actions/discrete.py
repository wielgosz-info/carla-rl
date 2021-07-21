import gym
import numpy as np
from carla import VehicleControl
from .actions_converter import ActionsConverter


class DiscreteActionsConverter(ActionsConverter):
    '''
        Actions Converter as in original CARLA RL model from driving benchmark:
        https://github.com/carla-simulator/reinforcement-learning/blob/master/agent/runnable_model.py
    '''

    def __init__(self, action_set: int = 13):
        '''


        :param action_set: 9 or 13, defaults to 13
        :type action_set: int, optional
        '''
        if action_set == 9:
            self.__discrete_actions = [[0., 0.], [-1., 0.], [-0.5, 0.], [0.5, 0.],
                                     [1.0, 0.], [0., -1.], [0., -0.5], [0., 0.5], [0., 1.]]
        else:
            self.__discrete_actions = [[0., 0.], [-1., 0.], [-0.5, 0.], [-0.25, 0.], [0.25, 0.], [0.5, 0.],
                                     [1.0, 0.], [0., -1.], [0., -0.5], [0., -0.25], [0., 0.25], [0., 0.5], [0., 1.]]

    def get_action_space(self):
        return gym.spaces.Discrete(len(self.__discrete_actions))

    def action_to_control(self, action, last_ego_vehicle_snapshot=None):
        control = VehicleControl()

        if isinstance(action, np.ndarray):
            action = action.item()
        if isinstance(action, int):
            print('Unexpected action got {}'.format(type(action)))

        action = self.__discrete_actions[action]
        if last_ego_vehicle_snapshot is not None and last_ego_vehicle_snapshot.get_velocity() * 3.6 < 30:
            control.throttle = action[0]
        elif action[0] > 0.:
            control.throttle = 0.
        control.steer = action[1]

        return control
