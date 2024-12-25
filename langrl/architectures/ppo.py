import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

# PPO Policy Network
class PPOPolicy(nn.Module):
    def __init__(self, state_dim, max_prompts, max_bots, max_routes):
        super(PPOPolicy, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.prompt_head = nn.Linear(256, max_prompts)
        self.bot_head = nn.Linear(256, max_bots)
        self.route_head = nn.Linear(256, max_routes)

    def forward(self, state):
        shared_output = self.shared_layers(state)
        prompt_logits = self.prompt_head(shared_output)
        bot_logits = self.bot_head(shared_output)
        route_logits = self.route_head(shared_output)
        return prompt_logits, bot_logits, route_logits


# PPO Agent
class PPOAgent:
    def __init__(
        self, 
        state_dim, 
        max_prompts, 
        max_bots, 
        max_routes, 
        batch_size = 256,
        learning_rate = 1e-3, 
        epsilon = 0.2, 
        gamma = 0.99,
        datapath = None,
        filepath=None, 
        modelpath=None
    ):
        self.max_prompts = max_prompts
        self.max_bots = max_bots
        self.max_routes = max_routes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.datapath = datapath
        self.filepath = filepath
        self.modelpath = modelpath
        
        # Ensure the file exists (or create it)
        if not os.path.exists(self.datapath):
            with open(self.datapath, "w") as f:
                pass  # Create the file if it doesn't exist
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy = PPOPolicy(state_dim, max_prompts, max_bots, max_routes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, active_prompts, active_bots, active_routes, temperature=0.01):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        prompt_logits, bot_logits, route_logits = self.policy(state)

        # Apply masks to logits
        prompt_mask = torch.tensor(
            [1 if i < active_prompts else 0 for i in range(self.max_prompts)], dtype=torch.float32).to(self.device)
        bot_mask = torch.tensor(
            [1 if i < active_bots else 0 for i in range(self.max_bots)], dtype=torch.float32).to(self.device)
        route_mask = torch.tensor(
            [1 if i < active_routes else 0 for i in range(self.max_routes)], dtype=torch.float32).to(self.device)
        
        # Scale logits by temperature
        prompt_logits = prompt_logits / temperature
        bot_logits = bot_logits / temperature
        route_logits = route_logits / temperature

        # Apply softmax only to active logits
        prompt_probs = torch.softmax(prompt_logits * prompt_mask, dim=-1)
        bot_probs = torch.softmax(bot_logits * bot_mask, dim=-1)
        route_probs = torch.softmax(route_logits * route_mask, dim=-1)

        # Sample actions
        prompt_action = torch.multinomial(prompt_probs, 1).item()
        bot_action = torch.multinomial(bot_probs, 1).item()
        route_action = torch.multinomial(route_probs, 1).item()

        # Calculate log probabilities for selected actions
        prompt_log_prob = torch.log(prompt_probs[prompt_action])
        bot_log_prob = torch.log(bot_probs[bot_action])
        route_log_prob = torch.log(route_probs[route_action])

        return (prompt_action, bot_action, route_action), (prompt_log_prob, bot_log_prob, route_log_prob)

    def train(self, transitions):
        states, actions, rewards, next_states, log_probs = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        advantages = self._compute_advantages(rewards)

        # Loss function for PPO
        self.optimizer.zero_grad()
        loss = self._compute_loss(states, actions, log_probs, advantages)
        loss.backward()
        self.optimizer.step()

    def _compute_advantages(self, rewards):
        advantages = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            advantages.insert(0, cumulative_reward)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _compute_loss(self, states, actions, log_probs_old, advantages):
        prompt_logits, bot_logits, route_logits = self.policy(states)

        # Extract the actions
        prompt_actions, bot_actions, route_actions = zip(*actions)

        # Compute log probabilities for the selected actions
        log_prob_prompts = torch.log_softmax(prompt_logits, dim=-1)
        log_prob_bots = torch.log_softmax(bot_logits, dim=-1)
        log_prob_routes = torch.log_softmax(route_logits, dim=-1)

        new_log_probs = (
            log_prob_prompts[range(len(prompt_actions)), prompt_actions] +
            log_prob_bots[range(len(bot_actions)), bot_actions] +
            log_prob_routes[range(len(route_actions)), route_actions]
        )

        # PPO clipped loss
        ratios = torch.exp(new_log_probs - log_probs_old)
        clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        return policy_loss
    
    def save_checkpoint(self, episode):
        checkpoint = {
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, self.filepath)
        print(f"Checkpoint saved to {self.filepath}")

    def load_checkpoint(self):
        if not torch.cuda.is_available():
            map_location = 'cpu'
        else:
            map_location = None

        try:
            checkpoint = torch.load(self.filepath, map_location=map_location)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            episode = checkpoint['episode']
            print(f"Checkpoint loaded from {self.filepath}. Resuming from episode {episode}.")
            return episode
        except FileNotFoundError:
            print(f"No checkpoint found at {self.filepath}. Starting from scratch.")
            return 0
        
    def save_model(self):
        torch.save(self.policy.state_dict(), self.modelpath)
        print(f"Trained model saved to {self.modelpath}")
    
    def load_model(self):
        try:
            self.policy.load_state_dict(torch.load(self.modelpath, map_location=self.device))
            print(f"Model loaded from {self.modelpath}")
        except FileNotFoundError:
            print(f"No model found at {self.modelpath}. Ensure the model has been trained and saved.")

    # Function to log data
    def log_experience(self, state, action, reward, next_state, log_probs):
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "log_probs": log_probs
        }
        with open(self.datapath, "a") as f:
            f.write(json.dumps(experience) + "\n")  # Append each experience as a JSON line
        
    # Load the logged data
    def load_experiences(self):
        experiences = []
        with open(self.datapath, "r") as f:
            for line in f:
                experiences.append(json.loads(line))
        
        return experiences

    # Training Loop
    def train_agent(self, env, num_episodes, active_prompts, active_bots, active_routes, steps=1):
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'log_prob'])
        temperature = 1.0  # Start with high temperature
        start_episode = self.load_checkpoint()
        num_episodes += start_episode
        transitions = []
        rewards = []
        
        scheduled_runs = num_episodes - start_episode
        if scheduled_runs < self.batch_size:
            raise ValueError(
                f"Scheduled runs ({scheduled_runs}) are less than the batch size ({self.batch_size}). exiting...")
            return rewards
        
        for episode in range(start_episode, num_episodes):
            state = env.reset()
            flat_state = env.flatten_state(state)   # Flatten state
            
            for t in range(steps):  # Max steps per episode
                # Gradually reduce temperature for less randomness
                temperature = max(0.1, 1.0 - episode / 100)  # Linear decay
                action, log_probs = self.select_action(
                    flat_state, active_prompts, active_bots, active_routes, temperature=temperature)
                next_state, reward, done, _ = env.step(action)
                transitions.append(Transition(flat_state, action, reward, next_state, sum(log_probs)))
                
                # For reporting
                rewards.append(reward)
                
                # Log the experience
                self.log_experience(flat_state, action, reward, next_state, sum(log_probs).item())
                
                state = next_state
                if done:
                    break
            
            if len(transitions) >= self.batch_size:
                self.train(transitions)
                transitions = []
            
            print(f"Episode {episode + 1}: Reward = {reward}, Temp. = {temperature:.2f}")
            
            # Save checkpoint every x episodes
            if (episode + 1) % self.batch_size == 0:
                self.save_checkpoint(episode + 1)

        if transitions:
            self.train(transitions)
            self.save_checkpoint(episode + 1)
            
        # After training
        self.save_model()
        
        return rewards

    def retrain_on_saved_data(self, num_episodes = None):
        Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'log_prob'])
        start_episode = self.load_checkpoint()
        experiences = self.load_experiences()
        transitions = []
        rewards = []
        
        if num_episodes is None:
            num_episodes = len(experiences) + start_episode
        else:
            num_episodes += start_episode
        
        for episode in range(start_episode, num_episodes):
            transitions.append(Transition(
                experiences[episode+start_episode]["state"], 
                experiences[episode+start_episode]["action"], 
                experiences[episode+start_episode]["reward"], 
                experiences[episode+start_episode]["next_state"], 
                torch.tensor(experiences[episode+start_episode]["log_probs"])
            ))
            
            # For reporting
            current_reward = experiences[episode+start_episode]["reward"]
            print(f"Episode {episode + 1}: Reward = {current_reward}")
            rewards.append(current_reward)
            
            if len(transitions) >= self.batch_size:
                self.train(transitions)
                transitions = []

            # Save checkpoint every 10 episodes
            if (episode + 1) % self.batch_size == 0:
                self.save_checkpoint(episode + 1)
        
        if transitions:
            self.train(transitions)
            self.save_checkpoint(episode + 1)
            
        # After training
        self.save_model()
        
        return rewards

    def inference(self, env, num_episodes=1, steps=1):
        # Load the model
        self.load_model()

        for episode in range(num_episodes):
            state = env.reset()
            flat_state = env.flatten_state(state)   # Flatten state
    
            total_reward = 0
            for t in range(steps):  # Max steps per episode
                action, log_probs = self.select_action(flat_state, env.active_prompts, env.active_bots, env.active_routes)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                # Log the experience
                self.log_experience(flat_state, action, reward, next_state, sum(log_probs).item())
    
                if done:
                    break
            print(f"Inference Episode {episode + 1}: Total Reward = {total_reward}")

