from collections import OrderedDict

import numpy as np
from gym_carla.converters.observations.observations import Observations
from typing import Dict
from carla import ActorSnapshot, Transform
from agents.navigation.local_planner import RoadOption
from agents.tools.misc import is_within_distance


class DirectionsObservations(Observations):
    # TODO: This class tries to follow what was done in
    # CarlaObservationsConverter in original code,
    # but should be updated to follow new RoadOption

    def __init__(self) -> None:
        super().__init__()

        self._bounds = OrderedDict(
            REACH_GOAL=[0, 1],
            GO_STRAIGHT=[0, 1],
            TURN_RIGHT=[0, 1],
            TURN_LEFT=[0, 1],
            LANE_FOLLOW=[0, 1],
        )

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        obs = np.zeros((5,), dtype=np.float32)

        # check if we should flag REACH_GOAL
        vehicle_transform = player_snapshot.get_transform()
        if is_within_distance(
            target.location,
            vehicle_transform.location,
            vehicle_transform.rotation,
            0.5,  # TODO: for now I set it to 0.5m (I'm assuming it's meters here...)
            5, -5  # allow some small angle variations ( +/-5deg). TODO: what values make sense here?
        ):
            obs[0] = 1.0
            return obs

        idx = {
            RoadOption.VOID: -1,
            RoadOption.LEFT: 3,
            RoadOption.RIGHT: 2,
            RoadOption.STRAIGHT: 1,
            RoadOption.LANEFOLLOW: 4,
            RoadOption.CHANGELANELEFT: -1,
            RoadOption.CHANGELANERIGHT: -1
        }[directions]

        if idx > -1:
            obs[idx] = 1.0

        return obs
