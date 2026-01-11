from typing import Tuple
import numpy as np


class RandomAgent:
    def __init__(self, observation_shape: Tuple[int, ...], num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def get_action(self, *args, **kwargs):
        # (1) TODO(student): Return a random action
        """
        The action space is discrete.
        Random from Discrete(N).
        """
        random_ac = np.random.randint(0, self.num_actions)
        return random_ac

    def update(self, *args, **kwargs):
        # Update is a no-op for the random agent
        return {}
