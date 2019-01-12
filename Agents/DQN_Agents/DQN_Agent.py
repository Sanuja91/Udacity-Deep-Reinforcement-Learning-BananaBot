

import os, sys, pathlib, torch, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Agents.Base_Agent import Base_Agent
from Utilities.Models.Neural_Network import Neural_Network
from Utilities.Data_Structures.Replay_Buffer import Replay_Buffer
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class DQN_Agent(Base_Agent):
    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"], config.seed)
        self.critic_local = Neural_Network(self.state_size, self.action_size, config.seed, self.hyperparameters, "VANILLA_NN").to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.hyperparameters["learning_rate"])

    def step(self):
        """Runs a step within a game including a learning step if required"""
        self.pick_and_conduct_action()
        self.update_next_state_reward_done_and_score()
        if self.time_for_critic_to_learn():
            self.critic_learn()
        self.save_experience()
        self.state = self.next_state #this is to set the state for the next iteration

    def pick_and_conduct_action(self):
        self.action = self.pick_action()
        self.conduct_action()

    def pick_action(self):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""

        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)

        self.critic_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.critic_local(state)
        self.critic_local.train() #puts network back in training mode

        action = self.make_epsilon_greedy_choice(action_values)
        return action

    def make_epsilon_greedy_choice(self, action_values):
        epsilon = self.hyperparameters["epsilon"] / (1.0 + (self.episode_number / self.hyperparameters["epsilon_decay_rate_denominator"]))

        if random.random() > epsilon:
            return np.argmax(action_values.data.cpu().numpy())
        return random.choice(np.arange(self.action_size))

    def critic_learn(self, experiences_given=False, experiences=None):

        if not experiences_given:
            states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_critic_optimisation_step(loss)

    def compute_loss(self, states, next_states, rewards, actions, dones):

        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        Q_targets_next = self.critic_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        Q_expected = self.critic_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def take_critic_optimisation_step(self, loss):

        if self.done: #we only update the learning rate at end of each episode
            self.update_learning_rate(self.hyperparameters["learning_rate"], self.critic_optimizer)

        self.critic_optimizer.zero_grad() #reset gradients to 0
        loss.backward() #this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.hyperparameters["gradient_clipping_norm"]) #clip gradients to help stabilise training
        self.critic_optimizer.step() #this applies the gradients

    def save_experience(self):
        self.memory.add_experience(self.state, self.action, self.reward, self.next_state, self.done)

    def locally_save_policy(self):
        pass
        # torch.save(self.qnetwork_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_critic_to_learn(self):
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        return self.episode_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def enough_experiences_to_learn_from(self):
        return len(self.memory) > self.hyperparameters["batch_size"]

    def sample_experiences(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones






























































# import numpy as np
# import random
# from collections import namedtuple, deque

# import torch
# import torch.nn.functional as F
# import torch.optim as optim

# import brains, constants

# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 64         # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR = 5e-4               # learning rate 
# UPDATE_EVERY = 4        # how often to update the network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class Agent():
#     """Interacts with and learns from the environment"""
    
#     def __init__(self, state_size, action_size, seed, nn_type):
#         """Intialize an Agent object
        
#         Params
#         =======
#         state_size(int): dimension of each state
#         action_size(int): dimension of each action
#         seed(int): random seed
#         """
        
#         self.state_size = state_size
#         self.action_size = action_size
#         self.seed = random.seed(seed)
#         valid_nn = False

#         if(nn_type == constants.VANILLA_DQN):
#             valid_nn = True
#             print("\nLoading Vanilla DQN\n")
           
#             # Vanilla DQN Network
#             self.qnetwork_local = brains.VanillaDQN(state_size, action_size, seed).to(device)
#             self.qnetwork_target = brains.VanillaDQN(state_size, action_size, seed).to(device)
            
#         if(valid_nn == False):
#             print("ERROR!!!! Invalid NN Architecture")
           

#         self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

#         # Replay memory
#         self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
#         # Initialize time step (for updating every UPDATE_EVERY steps)
#         self.t_step = 0
        
#     def step(self, state, action, reward, next_state, done):
#         # Save experience in replay memory
#         self.memory.add(state, action, reward, next_state, done)
        
#         # Learn every UPDATE_EVERY time steps
#         self.t_step = (self.t_step + 1) % UPDATE_EVERY
#         if self.t_step == 0:
#             # If enough samples are available in memory, get random subset and learn
#             if len(self.memory) > BATCH_SIZE:
#                 experiences = self.memory.sample()
#                 self.learn(experiences, GAMMA)
                
#     def act(self, state, eps=0.):
#         """Returns actions for the given state as per current policy
        
#         Params
#         =======
#             state (array_like): current state
#             eps (float): epsilon, for epsilon-greedy action selection
#         """
        
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         self.qnetwork_local.eval()
#         with torch.no_grad():
#             action_values = self.qnetwork_local(state)
#         self.qnetwork_local.train()
        
#         # Epsilon greedy action selection
#         if random.random() > eps:
#             return np.argmax(action_values.cpu().data.numpy())
#         else:
#             return random.choice(np.arange(self.action_size))
        
#     def learn(self, experiences, gamma):
#         """Update value parameters using given batch of experience tuples
        
#         Params
#         ========
#             experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
#             gamma (float): discount factor
#         """
#         states, actions, rewards, next_states, dones = experiences
        
#         # Get max predicted Q values (for next states) from target model
#         Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
#         # Compute Q targets for current states
#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
#         # Get expected Q values from local model
#         Q_expected = self.qnetwork_local(states).gather(1, actions)
        
#         # Compute loss
#         loss = F.mse_loss(Q_expected, Q_targets)
        
#         # Minimize the loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        
#         # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
#     def soft_update(self, local_model, target_model, tau):
#         """Soft update model parameters
#          θ_target = τ*θ_local + (1 - τ)*θ_target
         
#          Params
#          =======
#              local_model (PyTorch model): weights will be copied from
#              target_model (PyTorch model): weights will be copied to
#              tau (float): interpolation parameter
#         """
        
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


# class ReplayBuffer:
#     """Fixed size buffer to store experience tuples"""
    
#     def __init__(self, action_size, buffer_size, batch_size, seed):
#         """Initialize a ReplayBuffer object
        
#         Params
#         ========
#             action_size (int): dimension of each action
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#             seed (int): random seed
#         """
        
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.seed = random.seed(seed)
        
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory"""
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
        
#     def sample(self):
#         """Randomly sample a batch of experiences from memory"""
#         experiences = random.sample(self.memory, k=self.batch_size)
        
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)
