from collections import OrderedDict

import numpy as np
from gym_carla.converters.observations.observations import Observations
from typing import Dict
from carla import ActorSnapshot, Transform
from agents.navigation.local_planner import RoadOption


class TargetObservations(Observations):
    def __init__(self, rel_coord_system=False) -> None:
        super().__init__()

        self.__rel_coord_system = rel_coord_system

        if self.__rel_coord_system:
            self._bounds = OrderedDict(
                rel_x=[0, 100],
                rel_y=[0, 100],
                rel_x_unit=[0, 1],
                rel_y_unit=[0, 1]
            )
        else:
            self._bounds = OrderedDict(
                x=[0, 100],  # TODO: Fine-tune
                y=[0, 100],  # TODO: Fine-tune
                yaw=[0, 100]   # TODO: Fine-tune
            )

    def __get_relative_location_of_target(self, loc_x, loc_y, loc_yaw, target_x, target_y):
        veh_yaw = loc_yaw * np.pi / 180
        veh_dir_world = np.array([np.cos(veh_yaw), np.sin(veh_yaw)])
        veh_loc_world = np.array([loc_x, loc_y])
        target_loc_world = np.array([target_x, target_y])
        d_world = target_loc_world - veh_loc_world
        dot = np.dot(veh_dir_world, d_world)
        det = veh_dir_world[0]*d_world[1] - d_world[0]*veh_dir_world[1]
        rel_angle = np.arctan2(det, dot)
        target_location_rel_x = np.linalg.norm(d_world) * np.cos(rel_angle)
        target_location_rel_y = np.linalg.norm(d_world) * np.sin(rel_angle)

        return target_location_rel_x.item(), target_location_rel_y.item()

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        if self.__rel_coord_system:
            player_transform = player_snapshot.get_transform()
            target_rel_x, target_rel_y = self.__get_relative_location_of_target(
                player_transform.location.x,
                player_transform.location.y,
                player_transform.rotation.yaw,
                target.location.x,
                target.location.y,)
            target_rel_norm = np.linalg.norm(np.array([target_rel_x, target_rel_y]))
            target_rel_x_unit = target_rel_x / target_rel_norm
            target_rel_y_unit = target_rel_y / target_rel_norm

            return np.array([target_rel_x,
                             target_rel_y,
                             target_rel_x_unit,
                             target_rel_y_unit
                             ])
        else:
            return np.array([
                target.location.x,
                target.location.y,
                target.rotation.yaw
            ])
