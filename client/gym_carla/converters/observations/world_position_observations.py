from typing import Dict, OrderedDict

import numpy as np
from agents.navigation.local_planner import RoadOption
from carla import ActorSnapshot, Transform
from gym_carla.converters.observations.observations import Observations


class WorldPositionObservations(Observations):
    """
    Observations of player position in the world (x, y, and yaw).
    """

    def __init__(self) -> None:
        super().__init__()

        self._bounds = OrderedDict(
            x=[0, 100],  # location.x
            y=[0, 100],  # location.y
            yaw=[0, 1]   # rotation.yaw
        )

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        ego_transform = player_snapshot.get_transform()
        return np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.rotation.yaw
        ])
