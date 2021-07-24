import gym
import numpy as np
from PIL import Image as PILImage
from gym_carla.converters.observations.observations import Observations
from typing import Dict
from carla import ActorSnapshot, ColorConverter, Transform, Image
from agents.navigation.local_planner import RoadOption


class RGBCameraSensorException(Exception):
    def __init__(self, env_id):
        super().__init__()
        self.env_id = env_id


class RGBCameraSensorObservations(Observations):
    def __init__(self, h=84, w=84, sensor_id='rgb_camera') -> None:
        super().__init__()

        self.__h = h
        self.__w = w
        self.__c = 3
        self.__sensor_id = sensor_id

    def get_observation_space(self) -> gym.spaces.Space:
        img_shape = (self.__c, self.__w, self.__h)
        img_box = gym.spaces.Box(low=0, high=1, shape=img_shape, dtype=np.float32)
        return img_box

    def extract_observations(self,
                             player_snapshot: ActorSnapshot,
                             vehicle_sensors: Dict,
                             env_sensors: Dict,
                             directions: RoadOption,
                             target: Transform,
                             env_id) -> np.ndarray:
        data: Image = vehicle_sensors[self.__sensor_id]
        if data is not None:
            try:
                # convert image in-place
                data.convert(ColorConverter.Raw)

                img = PILImage.frombuffer('RGBA', (data.width, data.height), data.raw_data, "raw", 'RGBA', 0, 1)  # load
                img = img.resize((self.__w, self.__h), resample=PILImage.BICUBIC)             # resize
                img = img.convert('RGB')                                                      # drop alpha
                img = np.array(img)                                                           # convert to numpy array
                img = np.transpose(img, (2, 0, 1))                                            # [W,H,C] -> [C,W,H]
                return img
            except Exception as exc:
                raise RGBCameraSensorException(env_id) from exc
        else:
            return np.zeros((self.__c, self.__w, self.__h), dtype=np.float32)
