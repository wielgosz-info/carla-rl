from gym_carla.converters.observations.sensors.camera.rgb import RGBCameraSensorException
import random
import time
from typing import OrderedDict

import gym
import numpy as np
import queue

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import compute_distance
from carla import (Client, LaneChange, LaneMarkingType, Transform, VehicleControl,
                   WorldSettings, WorldSnapshot, command)

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
                 benchmark=False,
                 wait_for_data_timeout=1.0
                 ):

        self.logger = get_carla_logger()
        self.logger.info('Environment {} running in port {}'.format(env_id, port))

        self.host, self.port = host, port
        self.id = env_id

        self._obs_converter = obs_converter
        self.observation_space = self._obs_converter.get_observation_space()
        self._action_converter = action_converter
        self.action_space = self._action_converter.get_action_space()

        self._city_name = city_name

        # TODO: experiment suite should be a param (or a separate env if there is a lot of logic involved),
        # not this kind of weird switch...
        # or rather - benchmarking a suite is a different thing than using a suite for training,
        # but every suite can potentially be used in both settings
        if benchmark:
            self._experiment_suite = getattr(experiment_suites, exp_suite_name)(self._city_name)
        else:
            self._experiment_suite = getattr(experiment_suites, exp_suite_name)(self._city_name, subset)

        self._reward = getattr(rewards, reward_class_name)(self._obs_converter)
        self._experiments = self._experiment_suite.get_experiments()

        self._client = None
        self._world = None
        self._world_settings = WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=0.05,  # 20 FPS; fixed_delta_seconds <= max_substep_delta_time * max_substeps
            substepping=True,
            max_substep_delta_time=0.01,
            max_substeps=10
        )

        self._dao = None  # Needs to be reset after new world is loaded
        self._planner = None  # Needs DAO
        self._env_sensors = {}  # Sensors needed to manage env/generate observations
        self._vehicle_sensors = {}  # Sensors available to Vehicle
        self._sensors_buffer = {}
        self._wait_for_data_timeout = wait_for_data_timeout

        self._other_vehicles = []
        self._pedestrians = []
        self._ego_vehicle = None
        self._time_out = None
        self._episode_name = None
        self._target = None

        self._make_carla_client(host, port)

        blueprint_library = self._world.get_blueprint_library()
        self._vehicle_sensor_definitions = self._experiment_suite.prepare_sensors(blueprint_library)
        self._env_sensor_definitions = OrderedDict(
            collision=(blueprint_library.find('sensor.other.collision'), Transform()),
            lane_invasion=(blueprint_library.find('sensor.other.lane_invasion'), Transform())
        )

        self._distance_for_success = distance_for_success

        self.done = False
        self.last_obs = None
        self._initial_timestamp = None
        self.last_distance_to_goal = None
        self.last_direction = None
        self.last_snapshot = None
        self._success = False
        self._failure_timeout = False
        self._failure_collision = False
        self.benchmark = benchmark
        self.benchmark_index = [0, 0, 0]

        # TODO: should this be here?
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.steps = 0
        self.num_episodes = 0

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

        make_queue('world_snapshot', self._world.on_tick)

        return buffer

    def _get_data_from_buffer(self, key, frame):
        # based on https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py
        # but in case of event-based sensors there can be none or multiple data per frame
        try:
            while True:
                data = self._sensors_buffer[key].get(timeout=self._wait_for_data_timeout)
                if data.frame == frame:
                    # this was the only thing in queue
                    if self._sensors_buffer[key].empty():
                        return data
                    else:
                        seq = [data]
                        while not self._sensors_buffer[key].empty():
                            # this should never timeout
                            seq.append(self._sensors_buffer[key].get(timeout=self._wait_for_data_timeout))
                        return seq
        except queue.Empty:
            return None

    def _tick_the_world(self):
        # based on https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py
        frame = self._world.tick()

        # some sensors (like collision or lane_invasion) will return no data!
        return (
            self._get_data_from_buffer('world_snapshot', frame),
            {
                k: self._get_data_from_buffer(k, frame) for k in self._env_sensors.keys()
            },
            {
                k: self._get_data_from_buffer(k, frame) for k in self._vehicle_sensors.keys()
            },
        )

    def step(self, action):

        if self.done:
            raise ValueError('self.done should always be False when calling step')

        while True:

            try:
                # Send control
                control = self._action_converter.action_to_control(
                    action, self.last_snapshot.find(self._ego_vehicle.id))
                self._ego_vehicle.apply_control(control)

                # Move the simulation forward & get the observations
                world_snapshot, env_sensors_snapshot, vehicle_sensors_snapshot = self._tick_the_world()

                self.last_snapshot = world_snapshot
                current_timestamp = world_snapshot.timestamp

                distance_to_goal = self._get_distance_to_goal(world_snapshot, self._target)
                self.last_distance_to_goal = distance_to_goal

                directions = self._get_directions(world_snapshot,
                                                  self._target)
                self.last_direction = directions

                obs = self._obs_converter.convert(
                    world_snapshot.find(self._ego_vehicle.id),
                    vehicle_sensors_snapshot,
                    env_sensors_snapshot,
                    directions,
                    self._target,
                    self.id
                )

                self.last_obs = obs

            except RGBCameraSensorException:
                self.logger.debug('Camera Exception in step()')
                obs = self.last_obs
                distance_to_goal = self.last_distance_to_goal
                current_timestamp = self.last_snapshot.timestamp

            except RuntimeError as e:
                self.logger.debug('RuntimeError inside step(): {}'.format(e))
                self.done = True
                return self.last_obs, 0.0, True, {'carla-reward': 0.0}

            break

        # Check if terminal state
        timeout = (current_timestamp.elapsed_seconds - self._initial_timestamp.elapsed_seconds) > self._time_out
        collision, _ = self._is_collision(env_sensors_snapshot)
        success = distance_to_goal < self._distance_for_success
        if timeout:
            self.logger.debug('Timeout')
            self._failure_timeout = True
        if collision:
            self.logger.debug('Collision')
            self._failure_collision = True
        if success:
            self.logger.debug('Success')
        self.done = timeout or collision or success

        # Get the reward
        env_state = {'timeout': timeout, 'collision': collision, 'success': success}
        reward = self._reward.get_reward(self.last_snapshot, self._target, self.last_direction, control, env_state)

        # Additional information
        info = {'carla-reward': reward}

        self.steps += 1

        return obs, reward, self.done, info

    def reset(self):

        # Loop forever due to RuntimeErrors
        while True:
            try:
                self._reward.reset_reward()
                self.done = False

                self.close()

                if self.benchmark:
                    end_indicator = self._new_episode_benchmark()
                    if end_indicator is False:
                        return False
                else:
                    self._new_episode()

                # Hack: Try sleeping so that the server is ready. Reduces the number of TCPErrors
                # TODO: Possibly not needed now?
                time.sleep(4)

                # Move the simulation forward & get the observations
                world_snapshot, env_sensors_snapshot, vehicle_sensors_snapshot = self._tick_the_world()

                self._initial_timestamp = world_snapshot.timestamp
                self.last_snapshot = world_snapshot
                self.last_distance_to_goal = self._get_distance_to_goal(world_snapshot, self._target)

                directions = self._get_directions(world_snapshot, self._target)
                self.last_direction = directions

                obs = self._obs_converter.convert(
                    world_snapshot.find(self._ego_vehicle.id),
                    vehicle_sensors_snapshot,
                    env_sensors_snapshot,
                    directions,
                    self._target,
                    self.id
                )
                self.last_obs = obs
                self.done = False
                self._success = False
                self._failure_timeout = False
                self._failure_collision = False
                return obs

            except RGBCameraSensorException:
                self.logger.debug('Camera Exception in reset()')
                continue

            except RuntimeError as e:
                self.logger.debug('RuntimeError in reset()')
                self.logger.error(e)
                # Disconnect and reconnect
                self.close()
                time.sleep(5)
                self._make_carla_client(self.host, self.port)

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

            for i in range(0, len(self._pedestrians), 2):
                self._pedestrians[i].stop()
            if len(self._pedestrians):
                self._client.apply_batch([command.DestroyActor(x['id']) for x in self._pedestrians] +
                                         [command.DestroyActor(x['con']) for x in self._pedestrians])
        except RuntimeError as e:
            self.logger.debug('Error when destroying actors')
            self.logger.error(e)

        self._env_sensors = {}
        self._vehicle_sensors = {}
        self._ego_vehicle = None
        self._other_vehicles = []
        self._pedestrians = []

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
        self.logger.info('Env {} gets experiment {} with pose {}'.format(self.id, experiment_idx, idx_pose))

        self._new_episode_for_experiment(experiment, pose)

    def _new_episode_benchmark(self):
        experiment_idx_past = self.benchmark_index[0]
        pose_idx_past = self.benchmark_index[1]
        repetition_idx_past = self.benchmark_index[2]

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
                    self.benchmark_index = [experiment_idx_past + 1, 0, 1]
            else:
                experiment = experiment_past
                pose = poses_past[pose_idx_past + 1]
                self.benchmark_index = [experiment_idx_past, pose_idx_past + 1, 1]
        else:
            experiment = experiment_past
            pose = poses_past[pose_idx_past]
            self.benchmark_index = [experiment_idx_past, pose_idx_past, repetition_idx_past + 1]

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

        self._dao = GlobalRoutePlannerDAO(
            self._world.get_map(),
            0.2  # in meters
        )
        self._planner = GlobalRoutePlanner(self._dao)
        self._planner.setup()  # retrieve topology from server

        traffic_manager = self._client.get_trafficmanager()
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_random_device_seed(experiment.seed)
        traffic_manager.set_synchronous_mode(True)

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
        for sensor_id, (blueprint, transform) in self._env_sensor_definitions.items():
            self._env_sensors[sensor_id] = self._world.spawn_actor(
                blueprint, transform, attach_to=self._ego_vehicle)

        # attach vehicle sensors, as defined in experiment
        self._vehicle_sensors = {}
        for sensor_id, (blueprint, transform) in self._vehicle_sensor_definitions.items():
            self._vehicle_sensors[sensor_id] = self._world.spawn_actor(
                blueprint, transform, attach_to=self._ego_vehicle)

        # reset sensor buffers
        self._sensors_buffer = self._prepare_sensors_buffer()

        # add other vehicles according to experiment settings
        self._other_vehicles = self._spawn_other_vehicles(
            positions[0:start_index]+positions[start_index:],
            blueprints_vehicles,
            experiment.number_of_vehicles,
            traffic_manager
        )

        # add pedestrians according to experiment settings
        self._pedestrians = self._spawn_walkers(
            blueprints_pedestrians,
            experiment.number_of_pedestrians
        )

        self._time_out = self._experiment_suite.calculate_time_out(
            self._get_shortest_path(positions[start_index], positions[end_index]))
        self._target = positions[end_index]
        self._episode_name = str(experiment.task) + '_' + str(start_index) \
            + '_' + str(end_index)

        self.num_episodes += 1

    def _spawn_other_vehicles(self, positions, blueprints, number_of_vehicles, traffic_manager):
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
                self.logger.error(response.error)
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
                         .then(command.SetAutopilot(command.FutureActor, True, traffic_manager.get_port()))
                         )

        for response in self._client.apply_batch_sync(batch, True):
            if response.error:
                self.logger.error(response.error)
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
                self.logger.error(results[i].error)
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
                self.logger.error(results[i].error)
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
        directions = self._planner.abstract_route_plan(
            world_snapshot.find(self._ego_vehicle.id).get_transform().location,
            end_point.location
        )
        return directions[0]

    def _get_shortest_path(self, start_point, end_point):
        # TODO: check if 'route' will in fact contain waypoint-by-waypoint navigation
        # and we can use it to calculate distance
        route = self._planner.trace_route(start_point.location, end_point.location)
        return len(route) * self._dao.get_resolution()

    @ staticmethod
    def _is_collision(env_sensors_snapshot):

        collisions = []
        if env_sensors_snapshot['collision'] is not None:
            if isinstance(env_sensors_snapshot['collision'], list):
                collisions = env_sensors_snapshot['collision']
            else:
                collisions = [env_sensors_snapshot['collision']]

        invasions = []
        if env_sensors_snapshot['lane_invasion'] is not None:
            if isinstance(env_sensors_snapshot['lane_invasion'], list):
                invasions = [].extend(
                    [invasion.crossed_lane_markings for invasion in env_sensors_snapshot['lane_invasion']])
            else:
                invasions = env_sensors_snapshot['lane_invasion'].crossed_lane_markings

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
                self.logger.info("Trying to make client on port {}".format(port))
                self._client = Client(host, port)
                self._client.set_timeout(100)
                self._client.load_world(self._city_name, reset_settings=True)
                self._world = self._client.get_world()
                self.logger.info("Successfully made client on port {}".format(port))
                break
            except RuntimeError as error:
                self.logger.debug('Got RuntimeError... sleeping for 1')
                self.logger.error(error)
                time.sleep(1)
