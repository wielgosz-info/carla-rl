from gym_carla.converters.observations.observations_converter import ObservationsConverter


class Reward(object):
    def __init__(self, converter: ObservationsConverter):
        super().__init__()
        self.converter = converter

    def get_reward(self, world_snapshot, target, directions, action, env_state):
        raise NotImplementedError()

    def reset_reward(self):
        return
