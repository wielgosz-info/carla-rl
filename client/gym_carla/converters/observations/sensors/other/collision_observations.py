from collections import OrderedDict

import numpy as np
from gym_carla.converters.observations.observations import Observations
from typing import Dict, List, Union
from carla import ActorSnapshot, Transform, CollisionEvent
from agents.navigation.local_planner import RoadOption


class CollisionSensorObservations(Observations):
    def __init__(self) -> None:
        super().__init__()

        self._bounds = OrderedDict(
            vehicles=[0, 100],  # TODO: Fine-tune
            pedestrians=[0, 100],  # TODO: Fine-tune
            other=[0, 100],  # TODO: Fine-tune
        )

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        data: Union[CollisionEvent, List[CollisionEvent]] = env_sensors['collision']
        if data is not None:
            if isinstance(data, list):
                events = data
            else:
                events = [data]

            vehicles = np.sum([])
            pedestrians = np.sum([])
            other = np.sum([])

            return np.array([
                vehicles,
                pedestrians,
                other
            ])
        else:
            return np.array([0, 0, 0])
