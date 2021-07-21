from gym_carla.example_suites.corl_2017 import CoRL2017
from gym_carla.experiment_suite.experiment import Experiment


class TrainingSuite(CoRL2017):

    def __init__(self, city_name, subset=None):

        self._subset = subset
        super(TrainingSuite, self).__init__(city_name)

    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        if self._city_name == 'Town01':

            if self._subset == 'keep_lane':
                poses_tasks = [self._poses_town01()[0]]
                vehicles_tasks = [0]
                pedestrians_tasks = [0]

            elif self._subset == 'one_turn':
                poses_tasks = [self._poses_town01()[1]]
                vehicles_tasks = [0]
                pedestrians_tasks = [0]

            elif self._subset == 'keep_lane_one_turn':
                poses_tasks = self._poses_town01()[:2]
                vehicles_tasks = [0, 0]
                pedestrians_tasks = [0, 0]

            elif self._subset == 'no_dynamic_objects':
                poses_tasks = self._poses_town01()[:3]
                vehicles_tasks = [0, 0, 0]
                pedestrians_tasks = [0, 0, 0]

            elif self._subset is None:
                poses_tasks = self._poses_town01()
                vehicles_tasks = [0, 0, 0, 20]
                pedestrians_tasks = [0, 0, 0, 50]

            else:
                raise ValueError(
                    "experiments-subset must be keep_lane or keep_lane_one_turn or no_dynamic_objects or None")

        else:
            raise ValueError("city must be Town01 for training")

        experiments_vector = []

        for iteration in range(len(poses_tasks)):
            for weather in self.weathers:
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
                    repetitions=1
                )

                experiments_vector.append(experiment)

        return experiments_vector
