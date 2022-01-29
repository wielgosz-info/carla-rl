from agents.navigation.local_planner import RoadOption
from gym_carla.converters.observations.sensors.camera.rgb import RGBCameraSensorException
import random
import time
from typing import Any, Dict, List, OrderedDict, Tuple, Union

import gym
import numpy as np
import queue

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import compute_distance
from carla import (Client, LaneChange, LaneMarkingType, Transform, VehicleControl, SensorData,
                   WorldSettings, WorldSnapshot, ActorSnapshot, TrafficManager, World, command)

from gym_carla import rewards
from carla_logger import get_carla_logger

import gym_carla.example_suites as experiment_suites
from gym_carla.converters.actions.actions_converter import ActionsConverter
from gym_carla.converters.observations.observations_converter import \
    ObservationsConverter


class CarlaAVEnv(gym.Env):
    """
        An OpenAI Gym Environment for Autonomous Vehicles in CARLA.
        This environment expects RLAgents to return actions that can be directly mapped to VehicleControl.
        The environment that is going to utilize VehiclePIDContoller is planned for the future.
    """

    def __init__(self,
                 obs_converter: ObservationsConverter,
                 action_converter: ActionsConverter,
                 env_id,
                 random_seed=0,
                 exp_suite_name='TrainingSuite',
                 reward_class_name='SparseReward',
                 host='server',
                 port=2000,
                 city_name='Town01',
                 subset=None,
                 distance_for_success=2.0,
                 benchmark=False
                 ):

        self._logger = get_carla_logger()
        self._logger.info('Environment {} running in port {}'.format(env_id, port))

        self._env_id = env_id
        self._random_seed = random_seed

        self._obs_converter = obs_converter
        self._action_converter = action_converter

        self.observation_space = self._obs_converter.get_observation_space()
        self.action_space = self._action_converter.get_action_space()

        self._reward = getattr(rewards, reward_class_name)(self._obs_converter)

        self._host, self._port = host, port
        self._synchronous = True
        self._world_settings = WorldSettings(
            synchronous_mode=self._synchronous,
            fixed_delta_seconds=0.05,  # 10 FPS; fixed_delta_seconds <= max_substep_delta_time * max_substeps
            substepping=True,
            max_substep_delta_time=0.01,
            max_substeps=10
        )
        self._client: Client = None
        self._world: World = None
        self._traffic_manager: TrafficManager = None
        self._planner = None
        self._env_sensors = {}  # Sensors needed to manage env/generate observations
        self._vehicle_sensors = {}  # Sensors available to Vehicle

        self._sensors_buffer: OrderedDict[str, queue.Queue] = {}
        self._sensors_buffer_top: Dict[str, SensorData] = {}

        # TODO: experiment suite should be a param (or a separate env if there is a lot of logic involved),
        # not this kind of weird switch...
        # or rather - benchmarking a suite is a different thing than using a suite for training,
        # but every suite can potentially be used in both settings
        self._city_name = city_name
        self._episode_name = None
        if benchmark:
            self._experiment_suite = getattr(experiment_suites, exp_suite_name)(self._city_name)
        else:
            self._experiment_suite = getattr(experiment_suites, exp_suite_name)(self._city_name, subset)
        self._experiments = self._experiment_suite.get_experiments()

        self._other_vehicles = []
        self._pedestrians = []
        self._ego_vehicle = None

        self._time_out = None
        self._shortest_path = []
        self._target = None
        self._distance_for_success = distance_for_success

        self._make_carla_client(host, port)

        blueprint_library = self._world.get_blueprint_library()
        self._vehicle_sensors_definitions = self._experiment_suite.prepare_sensors(blueprint_library)
        self._env_sensors_definitions = OrderedDict(
            collision=(blueprint_library.find('sensor.other.collision'), Transform()),
            lane_invasion=(blueprint_library.find('sensor.other.lane_invasion'), Transform())
        )

        self._done = False
        self._last_obs = None
        self._initial_timestamp = None

        self._last_distance_to_goal = None
        self._last_direction = None
        self._last_world_snapshot = None
        self._last_env_sensors_snapshot = None
        self._last_vehicle_sensors_snapshot = None

        self._success = False
        self._failure_timeout = False
        self._failure_collision = False

        self._run_benchmark = benchmark
        self._benchmark_index = [0, 0, 0]

        # TODO: should this be here?
        np.random.seed(self._random_seed)
        random.seed(self._random_seed)

        self._steps = 0
        self._num_episodes = 0

    def _prepare_sensors_buffer(self):
        # based on https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py
        buffer = OrderedDict()

        def make_queue(key, register_event):
            data_queue = queue.Queue()
            register_event(data_queue.put)
            buffer[key] = data_queue

        for key, sensor in self._vehicle_sensors.items():
            make_queue(key, sensor.listen)

        for key, sensor in self._env_sensors.items():
            make_queue(key, sensor.listen)

        return buffer

    def _get_data_from_buffer(self, key, frame):
        '''
        Gets the data from queue.

        We handle sensors data the same no matter the sync/async mode,
        i.e. if the data is not available we skip it (without waiting).
        The observation converter should decide what to return
        when no data is available.
        '''
        data_seq = []

        if self._sensors_buffer_top[key] is not None:
            if self._sensors_buffer_top[key].frame <= frame:
                data_seq.append(self._sensors_buffer_top[key])
                self._sensors_buffer_top[key] = None
            else:
                return data_seq

        while not self._sensors_buffer[key].empty():
            data = self._sensors_buffer[key].get_nowait()
            if data.frame <= frame:
                data_seq.append(data)
            else:
                self._sensors_buffer_top[key] = data
                break

        return data_seq

    def _tick_the_world(self):
        if self._synchronous:
            frame = self._world.tick()
            world_snapshot = self._world.get_snapshot()
            assert world_snapshot.frame == frame
        else:
            world_snapshot = self._world.wait_for_tick()
            frame = world_snapshot.frame

        # some sensors (like collision or lane_invasion) will return no data!
        return (
            world_snapshot,
            {
                k: self._get_data_from_buffer(k, frame) for k in self._env_sensors.keys()
            },
            {
                k: self._get_data_from_buffer(k, frame) for k in self._vehicle_sensors.keys()
            },
        )

    def step(self, action):

        if self._done:
            raise ValueError('self.done should always be False when calling step')

        try:
            # Send control
            control = self._action_converter.action_to_control(
                action, self._last_world_snapshot.find(self._ego_vehicle.id))
            self._ego_vehicle.apply_control(control)

            # Move the simulation forward & get the observations
            (self._last_world_snapshot,
                self._last_env_sensors_snapshot,
                self._last_vehicle_sensors_snapshot) = self._tick_the_world()

            current_timestamp = self._last_world_snapshot.timestamp

            self._last_distance_to_goal = self._get_distance_to_goal(self._last_world_snapshot, self._target)

            self._last_direction = self._get_directions(self._last_world_snapshot,
                                                        self._target)

            obs = self._obs_converter.convert(
                self._last_world_snapshot.find(self._ego_vehicle.id),
                self._last_vehicle_sensors_snapshot,
                self._last_env_sensors_snapshot,
                self._last_direction,
                self._target,
                self._env_id
            )

            self._last_obs = obs

        except RuntimeError as e:
            self._logger.debug('RuntimeError inside step(): {}'.format(e))
            self._done = True
            return self._last_obs, 0.0, True, {'carla-reward': 0.0}

        # Check if terminal state
        timeout = (current_timestamp.elapsed_seconds - self._initial_timestamp.elapsed_seconds) > self._time_out
        collision, _ = self._is_collision_or_invasion(self._last_env_sensors_snapshot)
        success = self._last_distance_to_goal < self._distance_for_success
        if timeout:
            self._logger.debug('Timeout')
            self._failure_timeout = True
        if collision:
            self._logger.debug('Collision')
            self._failure_collision = True
        if success:
            self._logger.debug('Success')
        self._done = timeout or collision or success

        # Get the reward
        env_state = {'timeout': timeout, 'collision': collision, 'success': success}
        reward = self._reward.get_reward(self._last_world_snapshot, self._target,
                                         self._last_direction, control, env_state)

        # Additional information
        info = {'carla-reward': reward}

        self._steps += 1

        return obs, reward, self._done, info

    def reset(self):

        # Loop forever due to RuntimeErrors
        while True:
            try:
                self._reward.reset_reward()
                self._done = False

                self.close()

                if self._run_benchmark:
                    end_indicator = self._new_episode_benchmark()
                    if end_indicator is False:
                        return False
                else:
                    self._new_episode()

                # Move the simulation forward & get the observations
                (self._last_world_snapshot,
                 self._last_env_sensors_snapshot,
                 self._last_vehicle_sensors_snapshot) = self._tick_the_world()

                self._initial_timestamp = self._last_world_snapshot.timestamp
                self._last_distance_to_goal = self._get_distance_to_goal(self._last_world_snapshot, self._target)

                self._last_direction = self._get_directions(self._last_world_snapshot, self._target)

                self._last_obs = self._obs_converter.convert(
                    self._last_world_snapshot.find(self._ego_vehicle.id),
                    self._last_vehicle_sensors_snapshot,
                    self._last_env_sensors_snapshot,
                    self._last_direction,
                    self._target,
                    self._env_id
                )
                self._done = False
                self._success = False
                self._failure_timeout = False
                self._failure_collision = False

                return self._last_obs

            except RuntimeError as e:
                self._logger.debug('RuntimeError in reset()')
                self._logger.error(e)
                # Disconnect and reconnect
                self.close()
                time.sleep(5)
                self._make_carla_client(self._host, self._port)

    def close(self):
        try:
            for sensor in self._env_sensors.values():
                sensor.stop()
            self._client.apply_batch([command.DestroyActor(x.id) for x in self._env_sensors.values()])

            for sensor in self._vehicle_sensors.values():
                sensor.stop()
            self._client.apply_batch([command.DestroyActor(x.id) for x in self._vehicle_sensors.values()])

            vehicles_ids = self._other_vehicles
            if self._ego_vehicle is not None:
                vehicles_ids.append(self._ego_vehicle)
            if len(vehicles_ids):
                self._client.apply_batch([command.DestroyActor(x) for x in vehicles_ids])

            for controller in self._world.get_actors([p['con'] for p in self._pedestrians]):
                controller.stop()
            if len(self._pedestrians):
                self._client.apply_batch([command.DestroyActor(x['id']) for x in self._pedestrians] +
                                         [command.DestroyActor(x['con']) for x in self._pedestrians])
        except RuntimeError as e:
            self._logger.debug('Error when destroying actors')
            self._logger.error(e)

        self._env_sensors = {}
        self._vehicle_sensors = {}
        self._ego_vehicle = None
        self._other_vehicles = []
        self._pedestrians = []

        # TODO: this will probably need to be moved when running multi-agent
        if self._world:
            self._world.apply_settings(WorldSettings(synchronous_mode=False))
        if self._traffic_manager:
            self._traffic_manager.set_synchronous_mode(False)

    def render(self, mode, **kwargs):
        '''
        By default render nothing, all render modes should be handled by wrappers.
        '''
        pass

    def _get_distance_to_goal(self, world_snapshot: WorldSnapshot, target: Transform):
        distance_to_goal = compute_distance(
            world_snapshot.find(self._ego_vehicle.id).get_transform().location,
            target.location
        )
        return distance_to_goal

    def _new_episode(self):
        experiment_idx = np.random.randint(0, len(self._experiments))
        experiment = self._experiments[experiment_idx]
        idx_pose = np.random.randint(0, len(experiment.poses))
        pose = experiment.poses[idx_pose]
        self._logger.info('Env {} gets experiment {} with pose {}'.format(self._env_id, experiment_idx, idx_pose))

        self._new_episode_for_experiment(experiment, pose)

    def _new_episode_benchmark(self):
        experiment_idx_past = self._benchmark_index[0]
        pose_idx_past = self._benchmark_index[1]
        repetition_idx_past = self._benchmark_index[2]

        experiment_past = self._experiments[experiment_idx_past]
        poses_past = experiment_past.poses[0:]
        repetition_past = experiment_past.repetitions

        if repetition_idx_past == repetition_past:
            if pose_idx_past == len(poses_past) - 1:
                if experiment_idx_past == len(self._experiments) - 1:
                    return False
                else:
                    experiment = self._experiments[experiment_idx_past + 1]
                    pose = experiment.poses[0:][0]
                    self._benchmark_index = [experiment_idx_past + 1, 0, 1]
            else:
                experiment = experiment_past
                pose = poses_past[pose_idx_past + 1]
                self._benchmark_index = [experiment_idx_past, pose_idx_past + 1, 1]
        else:
            experiment = experiment_past
            pose = poses_past[pose_idx_past]
            self._benchmark_index = [experiment_idx_past, pose_idx_past, repetition_idx_past + 1]

        self._new_episode_for_experiment(experiment, pose)

    def _new_episode_for_experiment(self, experiment, pose):
        old_world_id = self._world.id
        self._client.load_world(experiment.map_name, reset_settings=True)

        self._world = self._client.get_world()
        while old_world_id == self._world.id:
            time.sleep(1)
            self._world = self._client.get_world()

        self._world.apply_settings(self._world_settings)
        self._world.set_weather(experiment.weather)

        self._planner = GlobalRoutePlanner(
            self._world.get_map(),
            0.2  # in meters
        )
        # self._planner.setup()  # retrieve topology from server

        positions = self._world.get_map().get_spawn_points()
        start_index = pose[0]
        end_index = pose[1]

        # spawn vehicles and walkers
        blueprint_library = self._world.get_blueprint_library()
        blueprints_vehicles = [x
                               for x in blueprint_library.filter("vehicle.*")
                               if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints_pedestrians = blueprint_library.filter("walker.pedestrian.*")

        # spawn ego vehicle
        blueprints_vehicles[0].set_attribute('role_name', 'ego')
        self._ego_vehicle = self._world.spawn_actor(
            blueprints_vehicles[0],
            positions[start_index]
        )
        self._ego_vehicle.apply_control(VehicleControl())

        # TODO: attach all of those sensors in batch via commands?
        self._env_sensors = {}
        for sensor_id, (blueprint, transform) in self._env_sensors_definitions.items():
            self._env_sensors[sensor_id] = self._world.spawn_actor(
                blueprint, transform, attach_to=self._ego_vehicle)

        # attach vehicle sensors, as defined in experiment
        self._vehicle_sensors = {}
        for sensor_id, (blueprint, transform) in self._vehicle_sensors_definitions.items():
            self._vehicle_sensors[sensor_id] = self._world.spawn_actor(
                blueprint, transform, attach_to=self._ego_vehicle)

        # reset sensor buffers
        self._sensors_buffer = self._prepare_sensors_buffer()
        self._sensors_buffer_top = {k: None for k in self._sensors_buffer.keys()}

        # add other vehicles according to experiment settings
        self._other_vehicles = self._spawn_other_vehicles(
            positions[0:start_index]+positions[start_index:],
            blueprints_vehicles,
            experiment.number_of_vehicles
        )

        # add pedestrians according to experiment settings
        self._pedestrians = self._spawn_walkers(
            blueprints_pedestrians,
            experiment.number_of_pedestrians
        )

        self._shortest_path = self._get_shortest_path(positions[start_index], positions[end_index])
        self._time_out = self._experiment_suite.calculate_time_out(
            len(self._shortest_path) * 0.2)  # 0.2 == sampling_resolution
        self._target = positions[end_index]
        self._episode_name = str(experiment.task) + '_' + str(start_index) \
            + '_' + str(end_index)

        self._num_episodes += 1

    def _spawn_other_vehicles(self, positions, blueprints, number_of_vehicles):
        """
        Spawn Vehicles

        From https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/spawn_npc.py,
        with lights removed for now
        """
        vehicles_list = []
        batch = []
        for transform in random.sample(positions, k=number_of_vehicles):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(command.SpawnActor(blueprint, transform).then(command.SetAutopilot(command.FutureActor, True)))

        for response in self._client.apply_batch_sync(batch):
            if response.error:
                self._logger.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        vehicles_list = []
        batch = []
        for transform in random.sample(positions, k=number_of_vehicles):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(command.SpawnActor(blueprint, transform)
                         .then(command.SetAutopilot(command.FutureActor, True, self._traffic_manager.get_port()))
                         )

        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                self._logger.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        return vehicles_list

    def _spawn_walkers(self,
                       blueprints,
                       number_of_walkers,
                       percentage_pedestrians_running=0.0,  # how many pedestrians will run
                       percentage_pedestrians_crossing=0.0  # how many pedestrians will walk through the road
                       ):
        """
        Spawn Walkers

        From https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/spawn_npc.py
        """

        all_id = []
        walkers_list = []

        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = Transform()
            loc = self._world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprints)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentage_pedestrians_running):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(command.SpawnActor(walker_bp, spawn_point))
        results = self._client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                self._logger.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self._world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(command.SpawnActor(walker_controller_bp, Transform(), walkers_list[i]["id"]))
        results = self._client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                self._logger.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self._world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self._world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self._world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self._world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        return walkers_list

    def _get_directions(self, world_snapshot: WorldSnapshot, end_point: Transform):
        directions = self._planner.trace_route(
            world_snapshot.find(self._ego_vehicle.id).get_transform().location,
            end_point.location
        )
        return directions[0][1]

    def _get_shortest_path(self, start_point, end_point):
        return self._planner.trace_route(start_point.location, end_point.location)

    @ staticmethod
    def _is_collision_or_invasion(env_sensors_snapshot):

        collisions = []
        if len(env_sensors_snapshot['collision']):
            collisions = env_sensors_snapshot['collision']

        invasions = []
        if len(env_sensors_snapshot['lane_invasion']):
            for inv in env_sensors_snapshot['lane_invasion']:
                invasions.extend(inv.crossed_lane_markings)

        # no way to get those in new CARLA?: intersection_offroad, intersection_otherlane
        # let just count the invasions... TODO: should LaneMarkingType.BrokenSolid count?
        invasions = [inv for inv in invasions
                     if inv.lane_change == LaneChange.NONE]
        invasions_curb_or_grass = [inv for inv in invasions
                                   if inv.type in [LaneMarkingType.Grass, LaneMarkingType.Curb]]
        invasions_solid = [inv for inv in invasions
                           if inv.type in [LaneMarkingType.Solid, LaneMarkingType.SolidSolid, LaneMarkingType.SolidBroken, LaneMarkingType.BrokenSolid]]

        return len(collisions) or len(invasions_curb_or_grass) or len(invasions_solid), collisions

    def _make_carla_client(self, host, port):
        while True:
            try:
                self._logger.info("Trying to make client on port {}".format(port))
                self._client = Client(host, port)
                self._client.set_timeout(10)
                self._client.load_world(self._city_name, reset_settings=True)
                self._world = self._client.get_world()
                self._traffic_manager = self._client.get_trafficmanager()
                # TODO: set_synchronous_mode needs to be set only in env that does the world tick!
                self._traffic_manager.set_synchronous_mode(self._synchronous)
                self._traffic_manager.set_hybrid_physics_mode(True)
                if self._synchronous:
                    self._traffic_manager.set_random_device_seed(self._random_seed)
                self._logger.info("Successfully made client on port {}".format(port))
                break
            except RuntimeError as error:
                self._logger.debug('Got RuntimeError... sleeping for 1')
                self._logger.error(error)
                time.sleep(1)

    @property
    def world_snapshot(self) -> WorldSnapshot:
        return self._last_world_snapshot

    @property
    def vehicle_sensors_types(self) -> OrderedDict[str, str]:
        return OrderedDict({k: b.type_id for k, (b, _) in self._vehicle_sensors_definitions})

    @property
    def vehicle_sensors_snapshot(self) -> Dict[str, Union[Any, List[Any]]]:
        return self._last_vehicle_sensors_snapshot

    @property
    def env_sensors_types(self) -> OrderedDict[str, str]:
        return OrderedDict({k: b.type_id for k, (b, _) in self._env_sensors_definitions})

    @property
    def env_sensors_snapshot(self) -> Dict[str, Union[Any, List[Any]]]:
        return self._last_env_sensors_snapshot

    @property
    def ego_vehicle_snapshot(self) -> ActorSnapshot:
        return self._last_world_snapshot.find(self._ego_vehicle.id)

    @property
    def shortest_path(self) -> List:
        return self._shortest_path

    @property
    def direction(self) -> RoadOption:
        return self._last_direction
