from typing import Any, Dict, Tuple, List
from functools import lru_cache

import gym
import numpy as np

from .carla_painter import CarlaPainter
from agents.navigation.local_planner import RoadOption
from carla import Waypoint


class CarlaVizRenderWrapper(gym.Wrapper):
    '''
    Wrapper to render the visualization with CarlaViz.

    Based on https://github.com/mjxu96/carlaviz/blob/master/examples/example.py
    '''

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.metadata['render.modes'].append('human')
        self.metadata['render.modes'] = list(set(self.metadata['render.modes']))

        self.__painter = CarlaPainter('viz', 8089)
        self.__car_trajectory = []

    def reset(self, **kwargs) -> gym.spaces.Space:
        self.__car_trajectory = []
        return super().reset(**kwargs)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        res = super().step(action)

        # remember all points for trajectories, even if render is only called sometimes
        ego_location = self.env.ego_vehicle_snapshot.get_transform().location
        self.__car_trajectory.append([ego_location.x, ego_location.y, ego_location.z])

        return res

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            trajectories = []

            # needs to be first to be below
            if len(self.env.shortest_path):
                trajectories.append({
                    "vertices": self.__shortest_path_to_trajectory(self.env.shortest_path),
                    "color": "#FFFFFF",
                    "width": 0.25
                })

            trajectories.append({
                "vertices": self.__car_trajectory,
                "color": "#00FF00",
                "width": 1.0
            })

            # draw trajectories
            # TODO: change those to dict with individual colors and widths when carlaviz is updated
            self.__painter.draw_polylines(trajectories)

            # draw the current direction that the vehicle received
            if self.env.direction:
                ego_location = self.env.ego_vehicle_snapshot.get_transform().location
                next_dir_str = "{:s}".format(RoadOption(self.env.direction).name)
                self.__painter.draw_texts([next_dir_str],
                                          [[ego_location.x, ego_location.y, ego_location.z + 10.0]], size=20)

        return self.env.render(mode, **kwargs)

    def close(self):
        super().close()
        self.__painter.close()

    def __shortest_path_to_trajectory(self, shortest_path: List[Tuple[Waypoint, RoadOption]]):
        return [[w[0].transform.location.x, w[0].transform.location.y, w[0].transform.location.z] for w in shortest_path]
