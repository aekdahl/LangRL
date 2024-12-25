import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# DQN architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.001):
        super(DQN, self).__init__()
        
        if not isinstance(learning_rate, float):
            print(learning_rate)
            
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # Output Q-values for all actions


class DQNAgent:
    def __init__(
        self, 
        state_size, 
        action_size = None, 
        num_prompts = 1, 
        num_roles = 1, 
        num_routes = 1, 
        gamma=0.95, 
        epsilon=1.0, 
        epsilon_decay=0.995, 
        epsilon_min=0.01, 
        learning_rate=0.001, 
        batch_size=64, 
        replay_buffer_size=10000,
        target_update_frequency = 10,
        filepath=None, 
        modelpath=None,
        verbose=False
    ):
        """
        Parameters
        - gamma: Discount factor
        - learning_rate: Learning rate for optimizer
        - batch_size: Batch size for replay buffer sampling
        - epsilon_decay: Epsilon decay for exploration/exploitation tradeoff
        - epsilon_min: Minimum epsilon value
        - replay_buffer_size: Maximum size of the replay buffer
        - target_update_frequency: How often to update the target network
        """
        self.verbose = verbose
        self.filepath = filepath
        self.modelpath = modelpath
        self.state_size = state_size
        self.num_prompts = num_prompts
        self.num_roles = num_roles
        self.num_routes = num_routes
        self.action_size = (num_prompts * num_roles * num_routes) if action_size is None else action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon if modelpath is None else -1  # Initial epsilon for exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = []  # Replay memory
        self.replay_buffer_size = replay_buffer_size
        self.target_update_frequency = target_update_frequency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, self.action_size, learning_rate).to(self.device)
        self.target_net = DQN(state_size, self.action_size, learning_rate).to(self.device)
        self.update_target_network()
        
        if modelpath:  # If a model path is provided, load the model
            self.load_model()

    def update_target_network(self):
        """Copy weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.replay_buffer_size:  # Limit memory size
            self.memory.pop(0)

    def act(self, state):
        """Selects an action based on the current policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()  # Exploit: choose the best action
        # return torch.argmax(q_values, dim=1).item()  # Exploit: choose the best action

    def replay(self):
        """Trains the network using experiences from replay memory."""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Target Q values
        #
        # ToDo: Should we use no_grad() here?
        #
        #with torch.no_grad():
        #    next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        #    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Update policy network
        loss = self.policy_net.criterion(current_q_values.squeeze(), target_q_values)
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        """
        Save the trained model weights to a file.

        Parameters:
        - modelpath (str): The file path where the model should be saved.
        """
        torch.save(self.policy_net.state_dict(), self.modelpath)
        print(f"Model saved successfully to {self.modelpath}.")

    def load_model(self):
        """
        Load the trained model weights into the policy network.

        Parameters:
        - modelpath (str): Path to the saved model file (e.g., "agents/models/dqn_trained_model.pth")
        
        ToDo: Should loaded models be set to evaluation mode?
        ToDo: Should we include map_location in load_state_dict?
        """
        try:
            # Load the state dictionary of the trained model
            self.policy_net.load_state_dict(torch.load(self.modelpath))
            #self.policy_net.load_state_dict(torch.load(self.modelpath, map_location=self.device))
            self.update_target_network()
            #self.policy_net.eval()  # Set the model to evaluation mode
            print(f"Model loaded successfully from {self.modelpath}.")
        except Exception as e:
            print(f"Failed to load the model: {e}")


    def save_checkpoint(self, episode):
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.policy_net.optimizer.state_dict(),
            'replay_buffer': self.memory, 
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.filepath)
        print(f"Checkpoint saved at episode {episode}")


    def load_checkpoint(self):
        if os.path.isfile(self.filepath):
            checkpoint = torch.load(self.filepath)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.policy_net.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.memory = checkpoint['replay_buffer']
            self.epsilon = checkpoint['epsilon']
            print(f"Checkpoint loaded. Resuming from episode {checkpoint['episode']}")
            return checkpoint['episode']
        else:
            print("No checkpoint found at given filepath.")
            return 0  # Start from scratch if no checkpoint is found

    # Train DQN agent with custom environment
    def train(self, env, episodes=1, steps=50, verbose=False, print_interval=10, resume=False, save=False):

        rewards = []
        completed = 0
        overall_completed = 0

        # Check if a checkpoint exists and resume training from that point
        if resume:
            start_episode = self.load_checkpoint()
        else:
            start_episode = 0

        for episode in range(start_episode, episodes):
            state = env.reset()
            last_state = state
            episode_reward = 0
            episode_rewards = []
            episode_states = []
            episode_actions = []
            did_not_complete = True

            for t in range(steps):  # Set max steps per episode
                action = self.act(state)  # Get action from DQN agent
                episode_states.append(state)
                episode_actions.append(action)

                next_state, reward, done = env.step(action)
                episode_rewards.append(reward)
                episode_reward += reward

                # Store experience in replay buffer
                self.remember(state, action, reward, next_state, done)

                last_state = state
                state = next_state

                if done:
                    completed += 1
                    did_not_complete = False
                    break
                    
            self.replay()

            #if did_not_complete and episode > episodes*0.9:
            if episode > episodes*1:
                print(f'Actions taken: {episode_actions}')
                print(f'Actions taken: {episode_states}')
                print(f'Last state: {[[int(i) for i in list(last_state)], env.current_process_step, episode_actions[-2]]}')
                print(f'This state: {[[int(i) for i in list(next_state)], env.current_process_step, episode_actions[-1]]}')
        
            # Update target network every few episodes
            if episode % self.target_update_frequency == 0:
                self.update_target_network()

            rewards.append(episode_reward)
            if episode % print_interval == 0:
                if resume:
                    self.save_checkpoint(episode)
                if save:
                    self.save_model()
                if verbose:
                    actions_counts = Counter(episode_actions)
                    print(f"Episode {episode}, Reward: {episode_reward}, Steps: {t}, Completed: {completed}")
                    overall_completed += completed
                    completed = 0

        print(f'Completions reached {overall_completed} out of {episodes}.')
        return rewards

    def inference(self, env, episodes=1, max_steps=50, tasks=None):

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            if tasks:
                tasks.update_workitem(episode)
                task_name = tasks.workitems[episode].get('Task name', tasks.workitems[episode].get('Task Name', ''))
                if self.verbose:
                    print('\n===\n')
                    print('Starting task: ', task_name)
                    print('\n===\n')
                    

            for t in range(max_steps):
                action = self.act(state)  # Use the trained agent to select actions
                next_state, reward, done = env.step(action)  # Take action in the environment
                total_reward += reward
                state = next_state

                if self.verbose:
                    print(f"Action: {action}, Episode {episode + 1} - Total Reward: {total_reward}")
                    
                if done:
                    break

        print("Evaluation completed.")
