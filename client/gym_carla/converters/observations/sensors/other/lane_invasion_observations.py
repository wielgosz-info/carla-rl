from collections import OrderedDict

import numpy as np
from gym_carla.converters.observations.observations import Observations
from typing import Dict, List, Union
from carla import ActorSnapshot, Transform, LaneInvasionEvent, LaneMarkingType, LaneChange
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
        data_seq: List[LaneInvasionEvent] = env_sensors['lane_invasion']
        if len(data_seq):
            invasions = []

            for inv in data_seq:
                invasions.extend(inv.crossed_lane_markings)

            invasions = [inv for inv in invasions
                         if inv.lane_change == LaneChange.NONE]

            otherlane = len([inv for inv in invasions
                             if inv.type in [LaneMarkingType.Solid, LaneMarkingType.SolidSolid, LaneMarkingType.SolidBroken, LaneMarkingType.BrokenSolid]])
            offroad = len([inv for inv in invasions
                           if inv.type in [LaneMarkingType.Grass, LaneMarkingType.Curb]])

            return np.array([
                otherlane,
                offroad
            ])
        else:
            return np.array([0, 0])
