import torch
import torch.optim as optim
import numpy as np

from collections import namedtuple

Rollout = namedtuple('Rollout', ['obs', 'actions', 'rewards', 'values', 'log_probs', 'done'])

def sample_action_and_log_prob(logits_list):
    """
    Sample each sub-action from the logits.
    Return:
      actions: list or tensor of shape (num_sub_actions,)
      log_probs: sum of log_probs of each sub-action
    """
    actions = []
    log_probs = 0
    for logits in logits_list:
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        actions.append(a)
        log_probs += dist.log_prob(a)
    return torch.stack(actions, dim=-1), log_probs

def compute_returns_advantages(rollouts, gamma=0.99, lam=0.95):
    """
    Basic GAE-lambda advantage calculation.
    rollouts is a list of Rollout objects.
    """
    # We'll do a simplified version assuming each rollout is one trajectory
    returns = []
    advantages = []
    
    # Flatten everything for simplicity
    all_obs = []
    all_actions = []
    all_values = []
    all_rewards = []
    all_log_probs = []
    all_dones = []

    for step in rollouts:
        all_obs.append(step.obs)
        all_actions.append(step.actions)
        all_rewards.append(step.rewards)
        all_values.append(step.values)
        all_log_probs.append(step.log_probs)
        all_dones.append(step.done)
    
    # Convert to tensors
    obs_t = torch.FloatTensor(all_obs)
    actions_t = torch.LongTensor(all_actions)
    rewards_t = torch.FloatTensor(all_rewards)
    values_t = torch.FloatTensor(all_values).squeeze(-1)
    dones_t = torch.FloatTensor(all_dones)
    log_probs_t = torch.FloatTensor(all_log_probs)
    
    advantages_t = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards_t))):
        delta = rewards_t[t] + gamma * next_value * (1 - dones_t[t]) - values_t[t]
        gae = delta + gamma * lam * (1 - dones_t[t]) * gae
        advantages_t.insert(0, gae)
        next_value = values_t[t].item()
    
    advantages_t = torch.stack(advantages_t)
    returns_t = advantages_t + values_t
    
    return obs_t, actions_t, returns_t.detach(), advantages_t.detach(), log_probs_t


def ppo_update(policy, optimizer, obs, actions, returns, advantages, old_log_probs,
               clip_range=0.2, value_coeff=0.5, entropy_coeff=0.01):
    
    logits_list, values = policy(obs)
    
    # Recompute log_probs under current policy
    # We need to separate each sub-action
    # actions shape: [batch_size, num_sub_actions]
    # logits_list[i] shape: [batch_size, action_dims[i]]
    
    batch_size, num_sub_actions = actions.shape
    new_log_probs = torch.zeros(batch_size)
    entropy_sum = 0.0
    
    for i, logits in enumerate(logits_list):
        dist = torch.distributions.Categorical(logits=logits)
        sub_act = actions[:, i]
        new_log_probs_i = dist.log_prob(sub_act)
        new_log_probs += new_log_probs_i
        
        # Entropy
        entropy_sum += dist.entropy().mean()
    
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = F.mse_loss(values.squeeze(-1), returns)
    
    # Total loss
    loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_sum
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), policy_loss.item(), value_loss.item(), entropy_sum.item()


def train_unified_ppo(env, policy, num_iterations=1000, rollout_steps=10, ppo_epochs=4):
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    for iteration in range(num_iterations):
        rollouts = []
        
        obs = env.reset()
        done = False
        for step in range(rollout_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            logits_list, value = policy(obs_t)
            
            actions_t, log_probs_t = sample_action_and_log_prob(logits_list)
            
            actions_np = actions_t.squeeze(0).numpy()  # shape (num_sub_actions,)
            
            next_obs, reward, done, info = env.step(actions_np)
            
            rollouts.append(
                Rollout(obs=obs,
                        actions=actions_np,
                        rewards=reward,
                        values=value.detach().numpy(),
                        log_probs=log_probs_t.item(),
                        done=float(done))
            )
            
            obs = next_obs
            if done:
                obs = env.reset()
        
        # Compute returns/advantages
        obs_t, actions_t, returns_t, adv_t, old_log_probs_t = compute_returns_advantages(rollouts)
        
        # PPO updates
        for epoch in range(ppo_epochs):
            loss, p_loss, v_loss, ent = ppo_update(
                policy, optimizer,
                obs_t, actions_t, returns_t, adv_t, old_log_probs_t
            )
        
        if iteration % 50 == 0:
            print(f"Iter {iteration}, Loss {loss:.3f}, Policy {p_loss:.3f}, Value {v_loss:.3f}, Ent {ent:.3f}")
