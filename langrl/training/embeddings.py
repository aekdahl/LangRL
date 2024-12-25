

class Embedding:
  
  def train_sac_embedding(self):
      # Hyperparams
      obs_dim = 4
      act_dim = 3
      env = EmbeddingEnv(obs_dim=obs_dim, act_dim=act_dim, max_steps=50)
  
      sac_agent = SACAgent(
          state_dim=obs_dim,
          action_dim=act_dim,
          hidden_dim=256,
          gamma=0.99,
          tau=0.005,
          alpha=0.2,
          lr=3e-4,
          buffer_size=100000,
          batch_size=64,
          device="cpu"  # or "cuda"
      )
  
      num_episodes = 200
      steps_per_episode = 50
      updates_per_step = 1
  
      for episode in range(num_episodes):
          state = env.reset()
          episode_reward = 0
  
          for step in range(steps_per_episode):
              # 1) Select action (embedding)
              action = sac_agent.select_action(state, eval_mode=False)
  
              # 2) Step environment
              next_state, reward, done, info = env.step(action)
              episode_reward += reward
  
              # 3) Store transition
              sac_agent.store_transition(state, action, reward, next_state, float(done))
  
              # 4) Update agent
              sac_agent.update(updates=updates_per_step)
  
              # Move to next
              state = next_state
              if done:
                  break
  
          print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
  
      print("Training complete.")