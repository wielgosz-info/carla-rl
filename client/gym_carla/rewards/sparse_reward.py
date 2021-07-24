from gym_carla.rewards.reward import Reward
import numpy as np


class SparseReward(Reward):
    '''
        Implict Reward Function defined by the benchmarking code. 1 is successful, 0 otherwise. 
    '''

    def get_reward(self, world_snapshot, target, directions, action, env_state):
        if env_state['success']:
            return 1
        return 0
