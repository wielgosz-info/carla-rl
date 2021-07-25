from typing import Any, Dict, Tuple

import gym
import numpy as np

from .carla_painter import CarlaPainter


class CarlaVizRenderWrapper(gym.Wrapper):
    '''
    Wrapper to render the visualization with CarlaViz.

    Based on https://github.com/mjxu96/carlaviz/blob/master/examples/example.py
    '''

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.metadata['render.modes'] = list(set(self.metadata['render.modes'].append('human')))

        self.__painter = CarlaPainter('viz', 8089)
        self.__trajectories = [[]]

    def reset(self, **kwargs) -> gym.spaces.Space:
        self.__trajectories = [[]]
        return super().reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        res = super().step(action)

        # remember all points for trajectories, even if render is only called sometimes
        ego_location = self.env.ego_vehicle_snapshot.get_transform().location
        self.__trajectories[0].append([ego_location.x, ego_location.y, ego_location.z])

        return res

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            # draw trajectories
            self.__painter.draw_polylines(self.__trajectories)

            # draw ego vehicle's velocity & location just above the ego vehicle
            ego_velocity = self.env.ego_vehicle_snapshot.get_velocity()
            ego_location = self.env.ego_vehicle_snapshot.get_transform().location
            velocity_str = "{:.2f}, ".format(ego_velocity.x) + "{:.2f}".format(ego_velocity.y) \
                + ", {:.2f}".format(ego_velocity.z)
            self.__painter.draw_texts([velocity_str],
                                      [[ego_location.x, ego_location.y, ego_location.z + 10.0]], size=20)

        return self.env.render(mode, **kwargs)
