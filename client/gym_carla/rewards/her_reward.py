from gym_carla.rewards.reward import Reward
import numpy as np


class HERReward(Reward):
    '''
        A modified variant of sparse rewards for training HER. Returns 500 when successful
        and otherwise the velocity of the vehicle.
    '''

    def get_reward(self, measurements, target, directions, action, env_state):
        v = min(25, measurements.player_measurements.forward_speed * 3.6) / 25
        if env_state['success']:
            return 500.0
        return v
