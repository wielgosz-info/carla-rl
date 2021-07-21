import gym
from carla import VehicleControl, ActorSnapshot


class ActionsConverter(object):
    def get_action_space(self) -> gym.spaces.Space:
        pass

    def action_to_control(self, action, last_ego_vehicle_snapshot: ActorSnapshot = None) -> VehicleControl:
        pass
