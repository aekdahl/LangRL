import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedHierarchicalPolicy(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_size=64):
        """
        action_dims: e.g. [5, num_llms+1, num_llms+1, num_prompts+1, num_routes+1]
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        
        # Common encoder
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Each component of the multi-discrete action gets its own head
        self.heads = nn.ModuleList()
        for dim in action_dims:
            self.heads.append(nn.Linear(hidden_size, dim))

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        # produce logits for each action component
        logits_list = []
        for head in self.heads:
            logits_list.append(head(x))
        return logits_list


class UnifiedHierarchicalActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_size=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        
        # Shared encoder
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Policy heads
        self.heads = nn.ModuleList()
        for dim in action_dims:
            self.heads.append(nn.Linear(hidden_size, dim))
        
        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        """
        For convenience, let's have forward() return both policy logits and value.
        obs: [batch_size, obs_dim]
        Returns:
          logits_list: list of [batch_size, action_dim] for each action component
          value: [batch_size, 1]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        logits_list = [head(x) for head in self.heads]
        value = self.value_head(x)
        
        return logits_list, value
