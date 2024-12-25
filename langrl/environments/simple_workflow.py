import gym
from gym import spaces
import numpy as np

class SimpleWorkflowEnv(gym.Env):
    """
    A toy environment with a 'workflow phase' in {0,1,2,3}, 
    and 4 discrete actions: 0=start, 1=review, 2=restart, 3=finalize.
    The agent aims to reach phase=3 with maximum reward.
    """

    def __init__(self, max_steps=20):
        super(SimpleWorkflowEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0..3
        # We'll represent the 'phase' as an integer {0..3}. 
        # That is our entire observation space (discrete).
        # For DQN, let's store it as a single integer or we can embed it in a 1D Box.
        self.observation_space = spaces.Discrete(4)

        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.phase = 0  # start in 'not started'
        self.step_count = 0
        return self.phase  # integer in {0..3}

    def step(self, action):
        """
        0 = start, 1 = review, 2 = restart, 3 = finalize
        We'll define some simple transitions & rewards.
        """
        self.step_count += 1
        reward = 0
        done = False

        if action == 0:  # start
            if self.phase == 0:
                self.phase = 1
                reward = 1
            else:
                reward = 0

        elif action == 1:  # review
            if self.phase == 1:
                self.phase = 2
                reward = 2
            else:
                reward = 0

        elif action == 2:  # restart
            # If we're in any phase > 0, we go back to phase=1
            if self.phase > 0:
                self.phase = 1
            # reward = 0

        elif action == 3:  # finalize
            if self.phase == 2:
                self.phase = 3
                reward = 10
                done = True  # finishing
            else:
                reward = -1  # penalize finalizing too early

        # Check if we've reached phase=3 or exceeded max steps
        if self.phase == 3:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        # next state is the new phase
        return self.phase, reward, done, {}
