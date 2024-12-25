import torch

from env import UnifiedHierarchicalEnv
from policies import UnifiedHierarchicalActorCritic
from training import train_unified_ppo

if __name__ == "__main__":
    env = UnifiedHierarchicalEnv(num_llms=3, num_prompts=5, num_routes=2, max_steps=6)
    action_dims = [5, (3+1), (3+1), (5+1), (2+1)]  # from env.action_space
    policy = UnifiedHierarchicalActorCritic(obs_dim=5, action_dims=action_dims, hidden_size=64)
    
    train_unified_ppo(env, policy, num_iterations=500, rollout_steps=5, ppo_epochs=3)
    
    torch.save(policy.state_dict(), "unified_policy.pt")
    print("Done training. Model saved to unified_policy.pt")
