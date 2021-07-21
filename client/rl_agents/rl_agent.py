class RLAgent(object):
    '''
        RL Agent; this is a different concept than CARLA Agent found in agents.navigation.agent:Agent and subclasses
    '''

    def get_value(self, obs, hidden_state, mask):
        raise NotImplementedError()

    def act(self, obs, hidden_states, masks):
        raise NotImplementedError()

    def update(self, rollouts):
        raise NotImplementedError()
