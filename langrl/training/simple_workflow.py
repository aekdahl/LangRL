from ..architectures.simple_dqn import DQNAgent
from ..environments.simple_workflow import SimpleWorkflowEnv

class SimpleWorkflow:
  def train(self, checkpoint_path = "../models/simple_dqn_checkpoint.pth"):
      
      env = SimpleWorkflowEnv(max_steps=20)
      agent = DQNAgent(
          state_dim=4,   # phase in {0..3}, so we do a 4-d one-hot
          action_dim=4,  # start, review, restart, finalize
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
      )
  
      # Optional: load existing checkpoint
      agent.load(checkpoint_path, load_replay_buffer=True)
  
      num_episodes = 200
      for episode in range(num_episodes):
          state = env.reset()
          total_reward = 0
          done = False
          steps = 0
          while not done:
              action = agent.select_action(state)
              next_state, reward, done, _ = env.step(action)
              total_reward += reward
  
              agent.store_transition(state, action, reward, next_state, float(done))
              agent.update()
  
              state = next_state
              steps += 1
  
          print(f"Episode {episode+1}, reward={total_reward:.1f}, eps={agent.epsilon:.2f}")
  
          # Every X episodes, save a checkpoint
          if (episode + 1) % 50 == 0:
              agent.save(checkpoint_path)
              agent.save_replay_buffer(checkpoint_path.replace(".pth", "_replay.pkl"))
  
      print("Done training.")
