import gym
import numpy as np
import cv2
from gym_carla.converters.observations.observations import Observations
from typing import Dict
from carla import ActorSnapshot, Transform
from agents.navigation.local_planner import RoadOption


class RGBCameraSensorException(Exception):
    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id


class RGBCameraSensorObservations(Observations):
    def __init__(self, h=84, w=84, id='rgb_camera') -> None:
        super().__init__()

        self.__h = h
        self.__w = w
        self.__c = 3
        self.__id = id

    def get_observation_space(self) -> gym.spaces.Space:
        img_shape = (self.__c, self.__h, self.__w)
        img_box = gym.spaces.Box(low=0, high=1, shape=img_shape, dtype=np.float32)
        return img_box

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        try:
            img = cv2.resize(vehicle_sensors[self.__id].raw_data, (self.__h, self.__w)) / 255.0
        except:
            raise RGBCameraSensorException(env_id)
        img = np.transpose(img, (2, 0, 1))
        return img
