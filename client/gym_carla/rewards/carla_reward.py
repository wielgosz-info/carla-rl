from gym_carla.rewards.reward import Reward
import numpy as np


class CarlaReward(Reward):
    '''
        Reward used for training in the original CARLA paper (http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)

        From the paper:
        "The reward is a weighted sum of five terms: distance traveled towards the goal d in km, speed v in
        km/h, collision damage c, intersection with the sidewalk s (between 0 and 1), and intersection with
        the opposite lane o (between 0 and 1)"

    '''

    def __init__(self, converter):
        super().__init__(converter)
        # Reward is calculated based on differences between timesteps, so need to
        # save the current state for the next reward.
        self.state = None

    def get_reward(self, world_snapshot, target, direction, action, env_state):

        # Distance towards goal (in km)
        d_x = world_snapshot.player_measurements.transform.location.x
        d_y = world_snapshot.player_measurements.transform.location.y
        d_z = world_snapshot.player_measurements.transform.location.z
        player_location = np.array([d_x, d_y, d_z])
        goal_location = np.array([target.location.x,
                                  target.location.y,
                                  target.location.z])
        d = np.linalg.norm(player_location - goal_location) / 1000

        # Speed
        v = world_snapshot.player_measurements.forward_speed * 3.6
        # Collision damage
        c_v = world_snapshot.player_measurements.collision_vehicles
        c_p = world_snapshot.player_measurements.collision_pedestrians
        c_o = world_snapshot.player_measurements.collision_other
        c = c_v + c_p + c_o

        # Intersection with sidewalk
        s = world_snapshot.player_measurements.intersection_offroad

        # Intersection with opposite lane
        o = world_snapshot.player_measurements.intersection_otherlane

        # Compute reward
        r = 0
        if self.state is not None:
            r += 1000 * (self.state['d'] - d)
            r += 0.05 * (v - self.state['v'])
            r -= 0.00002 * (c - self.state['c'])
            r -= -0.1 * float(s > 0.001)
            r -= 2 * (o - self.state['o'])

        # Update state
        new_state = {'d': d, 'v': v, 'c': c, 's': s, 'o': o,
                     'd_x': d_x, 'd_y': d_y, 'd_z': d_z,
                     'c_v': c_v, 'c_p': c_p, 'c_o': c_o}
        self.state = new_state

        return r

    def reset_reward(self):

        self.state = None
