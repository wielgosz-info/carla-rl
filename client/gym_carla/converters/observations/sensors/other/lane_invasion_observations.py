from collections import OrderedDict

import numpy as np
from gym_carla.converters.observations.observations import Observations
from typing import Dict, List, Union
from carla import ActorSnapshot, Transform, LaneInvasionEvent
from agents.navigation.local_planner import RoadOption


class LaneInvasionSensorObservations(Observations):
    def __init__(self) -> None:
        super().__init__()

        self._bounds = OrderedDict(
            otherlane=[0, 1],
            offroad=[0, 1],
        )

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        data: Union[LaneInvasionEvent, List[LaneInvasionEvent]] = env_sensors['lane_invasion']
        if data is not None:
            if isinstance(data, list):
                events = data
            else:
                events = [data]

            otherlane = np.sum([])
            offroad = np.sum([])

            return np.array([
                otherlane,
                offroad
            ])
        else:
            return np.array([0, 0])
