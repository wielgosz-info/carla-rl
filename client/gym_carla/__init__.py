from gym.envs.registration import register

register(
    id='carla-av-v0',
    entry_point='gym_carla.envs:CarlaAVEnv',
)
