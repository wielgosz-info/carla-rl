# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# -------------------------------------------------------------------------------
#
# This file is intended to provide the same functions as
# https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/benchmark_tools/experiment_suites/experiment_suite.py
# but working with CARLA 0.9.11 and gym

import abc
from collections import OrderedDict
from gym_carla.converters.observations.sensors.camera.rgb import RGBCameraSensorObservations
from carla import Transform, Location, Rotation


class ExperimentSuite(object):

    def __init__(self, city_name):

        self._city_name = city_name
        self._experiments = self.build_experiments()

    def calculate_time_out(self, path_distance):
        """
        Function to return the timeout, in seconds,
        that is calculated based on distance (in meters).
        """
        #  Originally, path distance was in map coordinates
        #  and I have no idea how it corresponded to meters.
        #  But now we will supply it in meters since that's
        #  what we can get from
        #  GlobalRoutePlanner.track_route() * waypoints resolution.
        #  Also, we're only really ever interested in seconds
        #  (not milliseconds as documented in the original file).

        #  So, assuming the path_distance is in meters,
        #  and the minimal sensible average velocity is 10km/h (~2.78 m/s),
        #  and we're adding 10s of "bonus" time (start/stop),
        #  and we want the result to be in seconds
        #  we get the exact same equation ;)
        return ((path_distance / 1000.0) / 10.0) * 3600.0 + 10.0

    def get_number_of_poses_task(self):
        """
            Get the number of poses a task have for this benchmark
        """

        # Warning: assumes that all tasks have the same size

        return len(self._experiments[0].poses)

    def get_number_of_reps_poses(self):
        """
            Get the number of poses a task have for this benchmark
        """

        # Warning: assumes that all poses have the same number of repetitions

        return self._experiments[0].repetitions

    def get_experiments(self):
        """
        Getter for the experiment set.
        """
        return self._experiments

    def prepare_sensors(self, blueprint_library):
        sensors = OrderedDict(
            rgb_camera=self._prepare_camera(blueprint_library)
        )
        return sensors

    def _prepare_camera(self, blueprint_library):
        blueprint_camera = blueprint_library.find('sensor.camera.rgb')
        blueprint_camera.set_attribute('image_size_x', '800')
        blueprint_camera.set_attribute('image_size_y', '600')
        blueprint_camera.set_attribute('fov', '100')
        blueprint_camera.set_attribute('sensor_tick', '0.1')

        transform_camera = Transform(
            location=Location(x=+2.0, y=0.0, z=1.4),
            rotation=Rotation(-15.0, 0, 0)
        )

        return (blueprint_camera, transform_camera)

    @property
    def weathers(self):
        weathers = set(self.train_weathers)
        weathers.update(self.test_weathers)
        return weathers

    @property
    def collision_as_failure(self):
        return False

    @property
    def traffic_light_as_failure(self):
        return False

    @abc.abstractmethod
    def build_experiments(self):
        """
        Returns a set of experiments to be evaluated
        Must be redefined in an inherited class.
        """

    @abc.abstractproperty
    def train_weathers(self):
        """
        Return the weathers that are considered as training conditions
        """

    @abc.abstractproperty
    def test_weathers(self):
        """
        Return the weathers that are considered as testing conditions
        """
