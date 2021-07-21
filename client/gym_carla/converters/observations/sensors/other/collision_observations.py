from collections import OrderedDict
from gym_carla.converters.observations.observations import Observations
from typing import Dict
from carla import ActorSnapshot, Transform
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
                             env_id):
        pass
