import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import pickle
from collections import deque, namedtuple


class DQNetwork(nn.Module):
    """
    Simple MLP to produce Q-values from a one-hot state representation.
    state_dim = 4, action_dim = 4 (4 possible workflow actions).
    """
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=32):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim=4,
        action_dim=4,
        hidden_dim=32,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
        update_freq=1,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        device="cpu"
    ):
        # Hyperparams
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        # Q-networks: policy & target
        self.policy_net = DQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        # Epsilon & step counters
        self.epsilon = epsilon_start
        self.action_dim = action_dim
        self.update_step = 0  # for scheduling epsilon & updating target net

    def select_action(self, state):
        """
        Epsilon-greedy action selection. 
        'state' is an integer 0..3 indicating the workflow phase.
        We'll convert it to a one-hot tensor.
        """
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.update_step / self.epsilon_decay)
        )

        if random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Greedy action from policy net
            state_vec = self._state_to_input(state).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_vec)
            action = torch.argmax(q_values, dim=1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append(
            self.Transition(state, action, reward, next_state, done)
        )

    def update(self):
        # Only update if enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        self.update_step += 1

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch_state = [t.state for t in batch]
        batch_action = [t.action for t in batch]
        batch_reward = [t.reward for t in batch]
        batch_next_state = [t.next_state for t in batch]
        batch_done = [t.done for t in batch]

        # Convert to tensors
        batch_state_t = torch.cat([self._state_to_input(s) for s in batch_state], dim=0).to(self.device)
        batch_action_t = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        batch_reward_t = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        batch_next_state_t = torch.cat([self._state_to_input(s) for s in batch_next_state], dim=0).to(self.device)
        batch_done_t = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        # Current Q
        q_values = self.policy_net(batch_state_t)
        q_a = q_values.gather(1, batch_action_t)  # Q(s,a)

        # Target Q (Double DQN style)
        with torch.no_grad():
            # Next best action from policy net
            next_q_values_policy = self.policy_net(batch_next_state_t)
            next_actions = torch.argmax(next_q_values_policy, dim=1, keepdim=True)

            # Q from target net
            target_q_values = self.target_net(batch_next_state_t)
            q_next = target_q_values.gather(1, next_actions)

            # If done, no future reward
            q_target = batch_reward_t + (1 - batch_done_t) * self.gamma * q_next

        loss = nn.MSELoss()(q_a, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target update
        if self.update_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _state_to_input(self, s):
        """
        Convert integer 's' in [0..3] to a one-hot vector shape [1,4].
        """
        x = torch.zeros((1,4))
        x[0, s] = 1.0
        return x

    # ===========================
    #  SAVE / LOAD CHECKPOINT
    # ===========================
    def save(self, filename):
        """
        Save model weights (policy, target) + optimizer state, 
        plus any relevant metadata like step counts, epsilon, etc.
        """
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update_step": self.update_step,
            "epsilon": self.epsilon
        }
        torch.save(checkpoint, filename)
        print(f"DQN checkpoint saved to {filename}")

    def load(self, filename, load_replay_buffer=False):
        """
        Load model parameters from 'filename'.
        If load_replay_buffer=True, also attempt to load buffer from a separate file.
        """
        if not os.path.exists(filename):
            print(f"No checkpoint found at {filename}")
            return

        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_step = checkpoint["update_step"]
        self.epsilon = checkpoint["epsilon"]

        print(f"DQN checkpoint loaded from {filename}")

        if load_replay_buffer:
            buffer_filename = filename.replace(".pth", "_replay.pkl")
            if os.path.exists(buffer_filename):
                with open(buffer_filename, "rb") as f:
                    data = pickle.load(f)
                self.replay_buffer = deque(data, maxlen=self.replay_buffer.maxlen)
                print(f"Replay buffer loaded from {buffer_filename}")
            else:
                print(f"No replay buffer file found at {buffer_filename}")

    def save_replay_buffer(self, filename):
        """
        Save replay buffer to a pickle file.
        """
        with open(filename, "wb") as f:
            pickle.dump(list(self.replay_buffer), f)
        print(f"DQN replay buffer saved to {filename}")

    def load_replay_buffer(self, filename):
        """
        Load replay buffer from a pickle file.
        """
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
            self.replay_buffer = deque(data, maxlen=self.replay_buffer.maxlen)
            print(f"DQN replay buffer loaded from {filename}")
        else:
            print(f"No replay buffer file found at {filename}")
