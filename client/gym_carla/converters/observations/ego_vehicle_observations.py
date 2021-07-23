from collections import OrderedDict

import numpy as np
from gym_carla.converters.observations.observations import Observations
from typing import Dict
from carla import ActorSnapshot, Transform, Vector3D
from agents.navigation.local_planner import RoadOption


class EgoVehicleObservations(Observations):
    def __init__(self) -> None:
        super().__init__()

        self._bounds = OrderedDict(
            forward_speed=[0, 30],
            acceleration_x=[-100, 100],
            acceleration_y=[-100, 100],
        )

    def __get_forward_speed(self, vehicle_transform: Transform, vehicle_velocity: Vector3D) -> float:
        """
        Calculate the vehicle forward speed since it was replaced by velocity Vector3D.
        From https://github.com/carla-simulator/carla/issues/355#issuecomment-477472667

        :param vehicle_transform: Current vehicle transform
        :type vehicle_transform: Transform
        :param vehicle_velocity: Current vehicle velocity
        :type vehicle_velocity: Vector3D
        :return: forward speed in m/s
        :rtype: float
        """
        yaw_global = np.radians(vehicle_transform.rotation.yaw)

        rotation_global = np.array([
            [np.sin(yaw_global),  np.cos(yaw_global)],
            [np.cos(yaw_global), -np.sin(yaw_global)]
        ])

        vehicle_velocity = np.array([vehicle_velocity.y, vehicle_velocity.x])
        velocity_local = rotation_global.T @ vehicle_velocity
        # TODO: which one? :D
        return velocity_local[0]

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        transform: Transform = player_snapshot.get_transform()
        velocity: Vector3D = player_snapshot.get_velocity()
        acceleration: Vector3D = player_snapshot.get_acceleration()

        return np.array([
            self.__get_forward_speed(transform, velocity),
            acceleration.x,
            acceleration.y
        ])
