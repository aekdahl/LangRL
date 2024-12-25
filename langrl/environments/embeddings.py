import gym
import numpy as np
from gym import spaces

class EmbeddingEnv(gym.Env):
    """
    A dummy environment to demonstrate how you might train a SAC agent
    that outputs an embedding (continuous vector). 

    - Observation: Randomly generated float vector of dimension obs_dim.
    - Action: A continuous embedding vector of dimension act_dim (the SAC agent's action).
    - Reward: Some contrived function of the embedding, purely for demonstration.
    """
    def __init__(self, obs_dim=4, act_dim=3, max_steps=50):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        
        # Observation space: let's say it's a Box in [-1,1] of shape (obs_dim,)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: the SAC agent will produce an embedding in R^act_dim
        # We won't clamp it here, so let's just define a large box
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        self.steps = 0
        # Random observation in [-1, 1]
        self.state = np.random.uniform(-1,1, size=(self.obs_dim,)).astype(np.float32)
        return self.state

    def step(self, action):
        """
        action: a continuous vector of size (act_dim,), i.e. the embedding.
        
        For demonstration:
        - We'll compute a silly reward that is higher if the embedding
          has a certain magnitude or direction, etc.
        """
        self.steps += 1
        
        # Example: reward = negative L2 distance from some "goal" embedding
        # Just a random "goal" for the sake of demonstration:
        goal = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        # If the env's act_dim is bigger, you'd adapt the logic
        # or define a random goal each reset.

        # If the dimension doesn't match, just min(len(goal), len(action)):
        dim = min(len(goal), len(action))
        
        # L2 distance
        diff = goal[:dim] - action[:dim]
        dist = np.linalg.norm(diff)
        reward = -dist  # negative distance => higher reward if closer to 'goal'
        
        # Maybe add a small bonus if magnitude is below some threshold, etc.
        # We'll skip that for now.
        
        # Next state
        self.state = np.random.uniform(-1,1, size=(self.obs_dim,)).astype(np.float32)
        
        done = (self.steps >= self.max_steps)
        return self.state, reward, done, {}
