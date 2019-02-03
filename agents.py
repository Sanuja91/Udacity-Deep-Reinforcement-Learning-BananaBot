import numpy as np
import random, os
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import brains, constants

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment"""
    
    def __init__(self, env_info, state_size, action_size, seed, nn_type, load_agent = False, doubleDQN = False, PER = False):
        """Intialize an Agent object
        
        Params
        =======
        state_size(int): dimension of each state
        action_size(int): dimension of each action
        seed(int): random seed
        nn_type(string): neural network name
        load_agent(boolean): boolean to load agent or not
        doubleDQN(boolean): boolean to use double DQN or not
        PER(boolean): boolean to use prioritized experience replay or not
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.nn_type = nn_type
        self.doubleDQN = doubleDQN
        self.PER = PER
        valid_nn = False

        if(nn_type == constants.VANILLA_DQN):
            valid_nn = True
            
            if (doubleDQN):
                print("\nLoading Double Vanilla DQN\n")
            else:
                print("\nLoading Vanilla DQN\n")
           
            # Vanilla DQN Network
            self.qnetwork_local = brains.VanillaDQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = brains.VanillaDQN(state_size, action_size, seed).to(device)
        elif(nn_type == constants.DUELING_DQN):
            valid_nn = True

            if (doubleDQN):
                print("\nLoading Double Dueling DQN\n")
            else:
                print("\nLoading Dueling DQN\n")

            # Dueling DQN Network
            self.qnetwork_local = brains.DuelingDQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = brains.DuelingDQN(state_size, action_size, seed).to(device)

        if(valid_nn == False):
            print("ERROR!!!! Invalid NN Architecture")

        if(doubleDQN):
            self.nn_type = constants.DOUBLE + " " + self.nn_type
            
        if(PER):
            self.nn_type = constants.PER + " " + self.nn_type

        if(load_agent):
            self.load_agent(self.nn_type, True) # Loads the agent that completed the challenge

        summary(self.qnetwork_local, input_size = env_info.vector_observations[0].shape)
           

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.PER)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        """Returns actions for the given state as per current policy
        
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # If in double DQN mode, then take average action from two networks
        if(self.doubleDQN):
            self.qnetwork_target.eval()
            with torch.no_grad():
                action_values += self.qnetwork_target(state)
                action_values /= 2
            self.qnetwork_target.train()

        
        # Epsilon greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples
        
        Params
        ========
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        self.save_agent(self.nn_type)
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters
         θ_target = τ*θ_local + (1 - τ)*θ_target
         
         Params
         =======
             local_model (PyTorch model): weights will be copied from
             target_model (PyTorch model): weights will be copied to
             tau (float): interpolation parameter
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # Save the checkpoint 
    def save_agent(self, fileName):
        checkpoint = {'state_dict': self.qnetwork_local.state_dict()}
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        fileName = 'checkpoints\\' + fileName + '.pth'
        # print("\nSaving checkpoint\n")
        torch.save(checkpoint, fileName)

    # Loads a pre-trained model from a checkpoint
    def load_agent(self, fileName, complete):
        print("\nLoading checkpoint\n")
        if (complete):
            fileName = "COMPLETE - " + fileName
        filepath = 'checkpoints\\' + fileName + '.pth'

        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
            self.qnetwork_local.load_state_dict(checkpoint['state_dict'])
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(fileName))

class ReplayBuffer:
    """Fixed size buffer to store experience tuples"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed, PER):
        """Initialize a ReplayBuffer object
        
        Params
        ========
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.PER = PER
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""

        # if (self.PER):
        #     e = self.experience(state, action, reward, next_state, done, priority)
        # else:
        #     e = self.experience(state, action, reward, next_state, done, None)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # if (self.PER):
        #     priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)
        #     return (states, actions, rewards, next_states, dones, priorities)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)