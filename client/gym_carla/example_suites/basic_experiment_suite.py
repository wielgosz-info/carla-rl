# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# -------------------------------------------------------------------------------
#
# This file is intended to provide the same functions as
# https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/benchmark_tools/experiment_suites/basic_experiment_suite.py
# but working with CARLA 0.9.11 and gym

from carla import WeatherParameters

from gym_carla.experiment_suite.experiment import Experiment
from gym_carla.experiment_suite.experiment_suite import ExperimentSuite


class BasicExperimentSuite(ExperimentSuite):

    @property
    def train_weathers(self):
        return [WeatherParameters.ClearNoon]

    @property
    def test_weathers(self):
        return [WeatherParameters.ClearNoon]

    def build_experiments(self):
        """
            Creates the whole set of experiment objects,
            The experiments created depends on the selected Town.
        """

        # We check the town, based on that we define the town related parameters
        # The size of the vector is related to the number of tasks, inside each
        # task there is also multiple poses ( start end, positions )

        if self._city_name == 'Town01':
            poses_tasks = [[[7, 3]], [[138, 17]], [[140, 134]], [[140, 134]]]
            vehicles_tasks = [0, 0, 0, 20]
            pedestrians_tasks = [0, 0, 0, 50]
        else:
            poses_tasks = [[[4, 2]], [[37, 76]], [[19, 66]], [[19, 66]]]
            vehicles_tasks = [0, 0, 0, 15]
            pedestrians_tasks = [0, 0, 0, 50]

        # Based on the parameters, creates a vector with experiment objects.
        experiments_vector = []
        for weather in self.weathers:
            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                experiment = Experiment()
                experiment.set(
                    map_name=self._city_name,
                    weather=weather,
                    poses=poses,
                    task=iteration,
                    number_of_vehicles=vehicles,
                    number_of_pedestrians=pedestrians,
                    repetitions=2
                )
                experiments_vector.append(experiment)

        return experiments_vector
