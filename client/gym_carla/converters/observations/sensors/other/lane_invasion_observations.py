from collections import OrderedDict
from gym_carla.converters.observations.observations import Observations
from typing import Dict
from carla import ActorSnapshot, Transform
from agents.navigation.local_planner import RoadOption


class LaneInvasionSensorObservations(Observations):
    def __init__(self) -> None:
        super().__init__()

        self._bounds = OrderedDict(
            otherlane=[0, 100],  # TODO: Fine-tune
            offroad=[0, 100],  # TODO: Fine-tune
        )

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id):
        pass
