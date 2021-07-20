# This file is intended to provide the same functions as
# https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/benchmark_tools/experiment.py
# but working with CARLA 0.9.11 and gym


from typing import List
from carla import WorldSettings, WeatherParameters


class Experiment(object):
    """
    Experiment defines a certain task, under conditions
    A task is associated with a set of poses, containing start and end pose.

    Conditions are associated with a carla Settings and describe the following:

    Number Of Vehicles
    Number Of Pedestrians
    Weather
    Random Seed for the Traffic Manager

    """

    def __init__(self):
        self._task: int = 0
        self._task_name: str = ''
        self._map_name: str = 'Town01'
        self._poses: List[List[int, int]] = [[]]
        self._weather: WeatherParameters = WeatherParameters.ClearNoon
        self._repetitions: int = 1
        self._number_of_vehicles: int = 20
        self._number_of_pedestrians: int = 30
        self._seed: int = 123456789

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError('Experiment: no key named %r' % key)
            setattr(self, key, value)

    @property
    def task(self):
        return self._task

    @property
    def task_name(self):
        return self._task_name

    @property
    def weather(self):
        return self._weather

    @property
    def poses(self):
        return self._poses

    @property
    def repetitions(self):
        return self._repetitions

    @property
    def number_of_vehicles(self):
        return self._number_of_vehicles

    @property
    def number_of_pedestrians(self):
        return self._number_of_pedestrians

    @property
    def map_name(self):
        return self._map_name

    @property
    def seed(self):
        return self._seed
