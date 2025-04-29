import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import time  # Add time module for tracking execution time

# Import new environment file
from dqn_cube_env import DQNRubiksEnv, Move

# Modify save directory to use saved_models folder in current directory
# SAVE_DIR = "/content/drive/My Drive/Spring 2025/DS3001 Reinforcement Learning/saved_models"
SAVE_DIR = "./saved_models"  # Use saved_models folder in current directory
os.makedirs(SAVE_DIR, exist_ok=True)

class DQNAgent:
    """
    Rubik's Cube solving agent using DQN algorithm
    """
    def __init__(self, state_dim, action_dim, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_min=0.1, 
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=64,
                 update_target_freq=20):
        # Algorithm hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Create Q-network and target network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        
        # Pre-compile networks to avoid first prediction delay
        print("Pre-compiling networks...")
        dummy_state = np.zeros((1, self.state_dim))
        self.q_network.predict(dummy_state, verbose=0)
        self.target_network.predict(dummy_state, verbose=0)
        
        self.update_target_network()  # Initialize target network weights
        
        # Training counter
        self.train_step_counter = 0
    
    def _build_model(self):
        """Build deep Q-network"""
        # Use functional API to create model, avoid Sequential model warnings
        inputs = keras.Input(shape=(self.state_dim,))
        x = keras.layers.Dense(1024, activation='relu')(inputs)
        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_dim, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def update_target_network(self):
        """Update target network weights to match current Q-network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy strategy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self):
        """Learn from experience replay buffer"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_dim))
        targets = np.zeros((self.batch_size, self.action_dim))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_network.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(t)
            
            states[i] = state
            targets[i] = target
        
        # Train network
        history = self.q_network.fit(states, targets, epochs=1, verbose=0)
        
        # Increment training counter and update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()
            
        return history.history['loss'][0]

def train_rubiks_agent(iterations=50, 
                       episodes_per_iteration=100,
                       max_steps=100, 
                       lr=1e-3, 
                       gamma=0.99,
                       initial_scramble=10,  # Use normal scramble depth
                       debug=False):
    """
    Train Rubik's Cube solving agent using DQN algorithm, save model checkpoints and output performance plots
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    state_dim = 480  # Use state dimension from new environment
    action_dim = len(Move)  # Action space size
    
    # If in debug mode, reduce training scale
    if debug:
        episodes_per_iteration = 10
        max_steps = 20
    
    print(f"Creating DQN agent, state dimension: {state_dim}, action dimension: {action_dim}")
    # Create DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=lr,
        gamma=gamma,
        buffer_size=50000 if not debug else 1000,
        batch_size=64 if not debug else 8
    )
    
    print(f"Creating Rubik's Cube environment...")
    # Create environment - use new DQNRubiksEnv
    env = DQNRubiksEnv()
    
    # Metrics for recording training progress
    all_rewards = []
    all_losses = []
    success_rates = []
    
    print(f"Starting training, total {iterations} iterations...")
    
    for iteration in range(1, iterations + 1):
        start_time = time.time()
        print(f"Iteration {iteration}/{iterations} ...")
        
        iteration_rewards = []
        iteration_success = 0
        iteration_losses = []
        
        for episode in range(episodes_per_iteration):
            # Reset environment and randomly scramble cube
            start_reset = time.time()
            state = env.reset(initial_scramble)
            print(f"  Episode {episode+1}: Environment reset took {time.time() - start_reset:.2f} seconds")
            
            episode_reward = 0
            solved = False
            
            for step in range(max_steps):
                # Choose action
                action = agent.choose_action(state)
                
                # Execute action - use new environment's step method
                next_state, reward, done, info = env.step(action)
                
                # Store experience in replay buffer
                agent.remember(state, action, reward, next_state, done)
                
                # Learn from replay buffer
                loss = agent.replay()
                if loss > 0:
                    iteration_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    solved = True
                    break
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Record this episode's reward and success
            iteration_rewards.append(episode_reward)
            if solved:
                iteration_success += 1
                
            # Print progress
            if (episode + 1) % 5 == 0 or episode == 0:  # Increase log frequency
                print(f"  Episode {episode + 1}/{episodes_per_iteration}, "
                      f"Epsilon: {agent.epsilon:.4f}, "
                      f"Mean Reward: {np.mean(iteration_rewards[-min(5, len(iteration_rewards)):]):.2f}, "
                      f"Memory samples: {len(agent.memory)}")
        
        # Ensure we have data before calculating statistics
        if iteration_rewards:
            # Calculate current iteration's average metrics
            avg_reward = np.mean(iteration_rewards)
            success_rate = iteration_success / max(1, len(iteration_rewards))
            
            all_rewards.append(avg_reward)
            success_rates.append(success_rate)
            all_losses.extend(iteration_losses)
            
            iter_time = time.time() - start_time
            print(f"  Iteration {iteration} completed, took {iter_time:.2f} seconds")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Success rate: {success_rate:.2%}")
            
            # Save model checkpoint
            model_file = os.path.join(SAVE_DIR, f"dqn_model_{timestamp}_it{iteration}.keras")
            agent.q_network.save(model_file)
            print(f"  Model saved to: {model_file}")
            
            # Generate intermediate plots every 5 iterations
            if iteration % 5 == 0 and all_rewards:
                plot_metrics(all_rewards, success_rates, all_losses, timestamp, iteration)
    
    # Generate final training metrics plots
    if all_rewards:
        plot_metrics(all_rewards, success_rates, all_losses, timestamp, iterations, final=True)

def plot_metrics(rewards, success_rates, losses, timestamp, iteration, final=False):
    """Plot and save training metrics"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot average rewards
    axs[0].plot(rewards, 'b-')
    axs[0].set_title('Average Reward per Iteration')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Average Reward')
    axs[0].grid(True)
    
    # Plot success rates
    axs[1].plot(success_rates, 'g-')
    axs[1].set_title('Cube Solving Success Rate per Iteration')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Success Rate')
    axs[1].set_ylim([0, 1])
    axs[1].grid(True)
    
    # Plot losses
    if losses:
        axs[2].plot(losses, 'r-', alpha=0.5)
        axs[2].set_title('Training Loss')
        axs[2].set_xlabel('Training Step')
        axs[2].set_ylabel('Loss')
        axs[2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_type = "final" if final else f"it{iteration}"
    plot_file = os.path.join(SAVE_DIR, f"dqn_training_metrics_{timestamp}_{plot_type}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"  Training plots saved to: {plot_file}")

if __name__ == "__main__":
    # Set debug mode parameters for faster testing
    debug_mode = True  # Change to True for faster testing

    # Train Rubik's Cube agent with default parameters
    train_rubiks_agent(
        iterations=3 if debug_mode else 50,         # Number of training iterations
        episodes_per_iteration=5 if debug_mode else 100, # Training episodes per iteration
        max_steps=20 if debug_mode else 50,         # Maximum steps per episode
        lr=0.001,                  # Learning rate
        gamma=0.99,                # Discount factor
        initial_scramble=3 if debug_mode else 10,   # Reduce scramble depth for faster testing
        debug=debug_mode           # Whether to enable debug mode
    ) 
