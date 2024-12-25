import gym
import numpy as np
from gym import spaces

class UnifiedHierarchicalEnv(gym.Env):
    """
    Toy environment to demonstrate a unified hierarchical RL approach.
    """

    def __init__(self, num_llms=5, num_prompts=5, num_routes=3, max_steps=10):
        super().__init__()
        self.num_llms = num_llms
        self.num_prompts = num_prompts
        self.num_routes = num_routes
        self.max_steps = max_steps
        
        # Observations:
        #   phase: which part of the workflow are we in (0=not started,1=in_progress,2=in_review,3=done)
        #   user_query_embedding: a small vector simulating the "context" (size=4 here, as an example)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Actions could be a structured set. We'll unify them into a single large discrete or multi-discrete space.
        # For simplicity, let's define:
        #   action[0]: workflow action (0=none,1=start,2=review,3=restart,4=finalize)
        #   action[1]: number_of_llms (0..num_llms)
        #   action[2]: which LLM (0..num_llms) - if number_of_llms=1
        #   action[3]: which prompt (0..num_prompts)
        #   action[4]: which route (0..num_routes)
        # In a real system you'd have a more advanced scheme (e.g., subsets, multiple LLMs, etc.).
        
        self.action_space = spaces.MultiDiscrete([
            5,                # workflow action
            self.num_llms+1,  # how many LLMs to use (0..num_llms)
            self.num_llms+1,  # which LLM (toy example: pick 1 LLM)
            self.num_prompts+1,
            self.num_routes+1
        ])
        
        self.reset()

    def reset(self):
        self.step_count = 0
        self.phase = 0  # 0=not started
        # Toy context embedding
        self.user_query_embedding = np.random.uniform(-1, 1, size=(4,))
        return self._get_obs()

    def _get_obs(self):
        # Let's embed the phase as 1-hot and combine with user_query_embedding
        phase_onehot = np.zeros((1, ), dtype=np.float32)
        phase_onehot[0] = self.phase
        obs = np.concatenate([phase_onehot, self.user_query_embedding]).astype(np.float32)
        return obs

    def step(self, action):
        """
        action: a list/array of 5 integers
        [workflow_act, num_llms, llm_idx, prompt_idx, route_idx]
        """
        self.step_count += 1
        workflow_act, num_llms, llm_idx, prompt_idx, route_idx = action
        
        reward = 0.0
        done = False
        
        # 1) Handle workflow action
        if workflow_act == 1:  # start
            self.phase = 1
        elif workflow_act == 2:  # review
            self.phase = 2
        elif workflow_act == 3:  # restart
            self.phase = 1
        elif workflow_act == 4:  # finalize
            self.phase = 3
            done = True
        
        # 2) Check if we used LLMs if needed
        if num_llms > 0:
            # we "selected" some LLM and prompt/route
            # This is a toy reward shaping:
            # +1 if we used a valid LLM and prompt in a correct phase (e.g. in_progress or in_review)
            if self.phase in [1, 2] and 0 <= llm_idx < self.num_llms and 0 <= prompt_idx < self.num_prompts:
                reward += 1.0
        
        # 3) Check if finalizing in the correct phase
        if done:
            # additional bonus for finalizing at the right time
            if self.phase == 3 and self.step_count < self.max_steps:
                reward += 2.0
        
        # 4) Check step limit
        if self.step_count >= self.max_steps:
            done = True
        
        obs = self._get_obs()
        return obs, reward, done, {}
