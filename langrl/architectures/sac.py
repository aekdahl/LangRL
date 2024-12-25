import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# =========================
# 1. NETWORK DEFINITIONS
# =========================

class Actor(nn.Module):
    """
    The SAC actor outputs a Gaussian distribution in R^action_dim.
    We'll sample the 'embedding' action from that distribution.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output layer for mean
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        # Output layer for log_std
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Log std bounds to avoid numerical issues
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        """
        Returns mean, log_std for a Gaussian in action space.
        """
        x = self.net(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        # Clamp log_std into [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """
        Samples an action (embedding) using the reparameterization trick:
        z = mu + eps * sigma
        Also returns the log_prob of that action (for SAC's policy loss).
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization
        eps = torch.randn_like(mu)
        z = mu + std * eps  # action sample
        # We'll compute log_prob under the Gaussian, including the correction
        # for Tanh squashing if you do that. Here, let's assume we do NOT
        # clamp or tanh the action. If you want to clamp your embedding, adapt accordingly.

        # log_prob of a Gaussian
        log_prob = (-0.5 * ((z - mu) / (std + 1e-8)).pow(2)
                    - log_std
                    - np.log(np.sqrt(2 * np.pi))).sum(dim=1, keepdim=True)
        return z, log_prob

class Critic(nn.Module):
    """
    A Q-network that takes (state, action) and outputs a scalar Q-value.
    We'll have two of these for SAC (Q1, Q2).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# =========================
# 2. SAC AGENT
# =========================

class SACAgent:
    """
    A minimal Soft Actor-Critic agent that outputs a continuous embedding as its action.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,         # Entropy coefficient (can be auto-tuned in advanced setups)
                 lr=3e-4,
                 buffer_size=100000,
                 batch_size=64,
                 device="cpu"):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        # Critics (Q1, Q2) and target critics
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)

        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        # Copy weights to target
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.Transition = namedtuple("Transition",
                                     ["state", "action", "reward", "next_state", "done"])

    def select_action(self, state, eval_mode=False):
        """
        state: np.array of shape (state_dim,)
        returns an np.array of shape (action_dim,) for your embedding
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if eval_mode:
            # In eval mode, just use mean (mu) for a deterministic policy
            with torch.no_grad():
                mu, log_std = self.actor(state_t)
                action = mu[0].cpu().numpy()
        else:
            with torch.no_grad():
                action_sample, _ = self.actor.sample(state_t)
            action = action_sample[0].cpu().numpy()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append(self.Transition(state, action, reward, next_state, done))

    def update(self, num_updates=1):
        """
        Perform 'num_updates' gradient steps of SAC using random mini-batches from replay.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data to train

        for _ in range(num_updates):
            batch = random.sample(self.replay_buffer, self.batch_size)
            batch_state = torch.FloatTensor([t.state for t in batch]).to(self.device)
            batch_action = torch.FloatTensor([t.action for t in batch]).to(self.device)
            batch_reward = torch.FloatTensor([t.reward for t in batch]).unsqueeze(-1).to(self.device)
            batch_next_state = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
            batch_done = torch.FloatTensor([t.done for t in batch]).unsqueeze(-1).to(self.device)

            # Critic update
            with torch.no_grad():
                # next action, log_prob from actor
                next_action, next_log_prob = self.actor.sample(batch_next_state)

                # next Q
                q1_next = self.critic1_target(batch_next_state, next_action)
                q2_next = self.critic2_target(batch_next_state, next_action)
                q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob

                # Bellman backup
                target_q = batch_reward + (1 - batch_done) * self.gamma * q_next

            # Q1, Q2 loss
            q1 = self.critic1(batch_state, batch_action)
            q2 = self.critic2(batch_state, batch_action)
            critic1_loss = nn.MSELoss()(q1, target_q)
            critic2_loss = nn.MSELoss()(q2, target_q)

            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            self.critic1_optim.step()

            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            self.critic2_optim.step()

            # Actor update
            new_action, log_prob = self.actor.sample(batch_state)
            q1_new = self.critic1(batch_state, new_action)
            q2_new = self.critic2(batch_state, new_action)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_prob - q_new).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Soft update of targets
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

    def soft_update(self, net, net_target):
        """
        Polyak averaging update: net_target = tau*net + (1-tau)*net_target
        """
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """
        Saves the model parameters (actor/critics/optimizers) 
        and optionally the replay buffer, to a .pth or similar file.
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optim.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optim.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optim.state_dict(),
            # If you want to store other metadata (e.g., total steps, episode count), add them here:
            # "total_steps": self.total_steps,
            # ...
        }

        # Save to disk
        torch.save(checkpoint, filename)
        print(f"SAC checkpoint saved to {filename}")

    def load(self, filename, load_replay_buffer=False):
        """
        Loads the model parameters from a checkpoint. 
        If load_replay_buffer=True, also load the replay buffer from a separate file or dict.
        """
        if not os.path.exists(filename):
            print(f"No checkpoint found at {filename}")
            return

        checkpoint = torch.load(filename, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])

        self.actor_optim.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic1_optim.load_state_dict(checkpoint["critic1_optimizer_state_dict"])
        self.critic2_optim.load_state_dict(checkpoint["critic2_optimizer_state_dict"])

        print(f"SAC checkpoint loaded from {filename}")

        if load_replay_buffer:
            # Optional: load replay buffer from a separate file 
            # (or from the same checkpoint if you store it there)
            buffer_filename = filename.replace(".pth", "_replay.pkl")
            if os.path.exists(buffer_filename):
                with open(buffer_filename, "rb") as f:
                    data = pickle.load(f)
                # Rebuild the deque with existing maxlen
                self.replay_buffer = deque(data, maxlen=self.replay_buffer.maxlen)
                print(f"Replay buffer loaded from {buffer_filename}")
            else:
                print(f"No replay buffer file found at {buffer_filename}")

        def save_replay_buffer(self, filename):
        # Convert the deque to a list and pickle it
        with open(filename, "wb") as f:
            pickle.dump(list(self.replay_buffer), f)

    def load_replay_buffer(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
            self.replay_buffer = deque(data, maxlen=self.replay_buffer.maxlen)
            print(f"Replay buffer loaded from {filename}")
        else:
            print(f"No replay buffer file found at {filename}")
