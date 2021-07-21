from gym_carla.converters.observations.observations import Observations
from typing import Dict, Union, List
from gym import spaces
import numpy as np
import cv2
import gym

from carla import ActorSnapshot, RoadOption, Transform


class ObservationsConverter(object):
    """
    Utility class to convert various configurable observations from CARLA
    to gym spaces. Right now aims to be compatible with original code,
    but this will probably change in he future.
    """

    def __init__(self, observations_items: Dict[str, Union[Observations, List[Observations]]]) -> None:
        """
        Creates ObservationsConverter.

        :param observations_items: Dictionary of (possibly nested) gym_carla.converters.observations.Observations
            objects. Each object should correspond to one of always available measurements (DirectionsObservations,
            EgoVehicleObservations, TargetObservations, WorldPositionObservations, CollisionSensorObservations,
            and LaneInvasionObservations) or match the vehicle-attached sensors as returned by
            ExperimentSuite.prepare_sensors method (it contains only 'sensors.camera.rgb' by default,
            which translates to gym_carla.converters.observations.sensors.camera.rgb.RGBCameraSensorObservations).

            Currently the nesting-and-flattening of the Observations assumes that all nested Observations
            return gym.spaces.Box.
        :type observations_items: Dict[str, Union[Observations, List[Observations]]]
        """
        super().__init__()

        self.__observations = observations_items

    def convert(self,
                player_snapshot: ActorSnapshot,
                vehicle_sensors: Dict,
                env_sensors: Dict,
                directions: RoadOption,
                target: Transform,
                env_id) -> Dict[str, np.ndarray]:

        out = {}
        for key, observations in self.__observations.items():
            if isinstance(observations, (list, tuple)):
                out[key] = np.concatenate([
                    o.extract_observations(
                        player_snapshot,
                        vehicle_sensors,
                        env_sensors,
                        directions,
                        target,
                        env_id
                    ) for o in observations
                ])
            else:
                out[key] = observations.extract_observations(
                    player_snapshot,
                    vehicle_sensors,
                    env_sensors,
                    directions,
                    target,
                    env_id
                )
        return out

    def get_observation_space(self) -> gym.spaces.Dict:
        out = {}
        for key, observations in self.__observations.items():
            if isinstance(observations, (list, tuple)):
                nested_spaces = [(s.low, s.high) for s in [
                    o.get_observation_space() for o in observations
                ]]
                low, high = zip(*nested_spaces)
                out[key] = gym.spaces.Box(
                    low=np.concatenate(low),
                    high=np.concatenate(high)
                )
            else:
                out[key] = observations.get_observation_space()
        return gym.spaces.Dict(out)
