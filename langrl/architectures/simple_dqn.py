import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

class DQNetwork(nn.Module):
    """
    A simple MLP that takes as input: state (phase in [0..3]) 
    and outputs Q-values for 4 possible actions.
    We'll embed the phase as a one-hot or just an integer -> embedding.
    """
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=32):
        super(DQNetwork, self).__init__()
        # state_dim=4 is a bit silly, but let's do a small net.
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
        update_freq=10,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        device="cpu"
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        # Q networks
        self.policy_net = DQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        self.epsilon = epsilon_start
        self.action_dim = action_dim
        self.update_step = 0  # count how many steps we've updated

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        state is just an integer {0..3}, we'll one-hot it so the NN sees a 4D input
        """
        # Epsilon decay
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.update_step / self.epsilon_decay)
        )

        if random.random() < self.epsilon:
            # random action
            return random.randint(0, self.action_dim - 1)
        else:
            # use policy
            state_vec = self._state_to_input(state)  # shape [1,4]
            with torch.no_grad():
                q_values = self.policy_net(state_vec)
            action = torch.argmax(q_values, dim=1).item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append(
            self.Transition(state, action, reward, next_state, done)
        )

    def update(self):
        # only update if enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        self.update_step += 1

        # sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        # convert to tensors
        batch_state = [t.state for t in batch]
        batch_action = [t.action for t in batch]
        batch_reward = [t.reward for t in batch]
        batch_next_state = [t.next_state for t in batch]
        batch_done = [t.done for t in batch]

        batch_state_t = torch.cat([self._state_to_input(s) for s in batch_state], dim=0).to(self.device)
        batch_action_t = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        batch_reward_t = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        batch_next_state_t = torch.cat([self._state_to_input(s) for s in batch_next_state], dim=0).to(self.device)
        batch_done_t = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        # current Q
        q_values = self.policy_net(batch_state_t)
        q_a = q_values.gather(1, batch_action_t)  # Q(s,a)

        # target Q
        with torch.no_grad():
            # Double DQN approach (optional):
            # next_action = argmax(policy_net(next_state))
            next_q_values = self.policy_net(batch_next_state_t)
            next_action = torch.argmax(next_q_values, dim=1, keepdim=True)
            # Q_next = target_net(s', a')
            target_q_values = self.target_net(batch_next_state_t)
            q_next = target_q_values.gather(1, next_action)

            # If done, no future reward
            q_target = batch_reward_t + (1 - batch_done_t) * self.gamma * q_next

        loss = nn.MSELoss()(q_a, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target net
        if self.update_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _state_to_input(self, s):
        """
        Convert integer phase s in {0..3} to a one-hot vector shape [1,4].
        """
        x = torch.zeros((1,4))
        x[0, s] = 1.0
        return x

