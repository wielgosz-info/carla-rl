from gym_carla.rewards.reward import Reward
import numpy as np


class CIRLReward(Reward):
    '''
        Reward function from https://arxiv.org/abs/1807.03776. 
    '''

    def _r_s(self, control, direction):
        if direction == 'TURN_RIGHT' and control.steer < 0:
            return -15
        if direction == 'TURN_LEFT' and control.steer > 0:
            return -15
        if direction == 'GO_STRAIGHT' and np.abs(control.steer) > 0.2:
            return -20
        return 0

    def _r_v(self, velocity, direction):
        if direction == 'LANE_FOLLOW':
            return min(25.0, velocity)
        if direction == 'GO_STRAIGHT':
            return min(35.0, velocity)
        if velocity <= 20:
            return velocity
        if velocity > 20:
            return 20 - velocity

        assert False

    def get_reward(self, world_snapshot, target, direction, control, env_state):

        reward = 0
        direction = self.converter.direction_to_string(direction)

        if direction == 'REACH_GOAL':
            # TODO: What is right to return here?
            return 0

        # Speed (km/h)
        v = world_snapshot.player_measurements.forward_speed * 3.6
        reward += self._r_v(v, direction)

        reward += self._r_s(control, direction)

        # Collisions (r_d in the paper)
        if world_snapshot.player_measurements.collision_vehicles > 1e-6 or world_snapshot.player_measurements.collision_pedestrians > 1e-6:
            reward += -100
        if world_snapshot.player_measurements.collision_other:
            reward += -50

        # Intersection with sidewalk (r_r)
        s = world_snapshot.player_measurements.intersection_offroad
        if s > 1e-6:
            reward += -100

        # Intersection with opposite lane (r_o)
        o = world_snapshot.player_measurements.intersection_otherlane
        if o > 1e-6:
            reward += -100

        return reward
