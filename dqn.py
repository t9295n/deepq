import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    '''
    Parameters for neural network
    Inputs
    - learning rate
    - number of states
    - number of actions
    - nodes in 1st fully connected hidden layer
    - nodes in 2nd fully connected hidden layer
    '''
    def __init__(self, learning_rate, nstates, nactions, nodes_fc1=256, nodes_fc2=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(nstates, nodes_fc1)
        self.fc2 = nn.Linear(nodes_fc1, nodes_fc2)
        self.fc3 = nn.Linear(nodes_fc2, nactions)
        self.activation = nn.ReLU()
    
    def forward(self, state):
        state = self.activation(self.fc1(state))
        state = self.activation(self.fc2(state))
        actions = self.fc3(state)
        return actions


class Agent():
    def __init__(self, gamma, learning_rate, nstates, nactions, batch_size, nodes=256):
        self.nstates = nstates
        self.nactions = nactions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss()

        self.Q_local = DQN(learning_rate, nstates, nactions, nodes_fc1=nodes, nodes_fc2=nodes).to(self.device)
        self.optimizer = optim.Adam(self.Q_local.parameters(), lr=learning_rate)

        # replay buffer
        self.index = 0
        self.memsize = int(1e6)    
        self.states = torch.zeros((self.memsize, nstates))
        self.next_states = torch.zeros((self.memsize, nstates))
        self.actions = torch.zeros(self.memsize)
        self.rewards = torch.zeros(self.memsize)
        self.dones = torch.zeros(self.memsize)
        
        self.steps = 0
        self.update_freq = 8
        
    def add(self, state, action, reward, next_state, done):
        '''
        Add experience to replay buffer
        '''
        i = self.index % self.memsize ## wrap around if full
        self.states[i] = torch.from_numpy(state).to(self.device)
        self.next_states[i] = torch.from_numpy(next_state).to(self.device)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.index += 1
        
    def sample(self):
        '''
        Sample from replay buffer
        '''      
        batch = np.random.choice(min(self.index, self.memsize), self.batch_size, replace=False)        
        states = self.states[batch]
        next_states = self.next_states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        dones = self.dones[batch]
        return (states, next_states, actions, rewards, dones)

    def policy(self, state, epsilon):        
        if np.random.random() < epsilon:
            return np.random.randint(self.nactions)
        else:
            state = torch.tensor(np.array(state)).to(self.device)
            self.Q_local.eval()
            with torch.no_grad(): 
                actions = self.Q_local(state)
            self.Q_local.train()
            return torch.argmax(actions).item()

    def optimize(self):
        self.steps += 1
        if (self.steps % self.update_freq == 0) & (self.index > self.batch_size):
            ## sample, get q-vals
            states, next_states, actions, rewards, dones = self.sample()
            self.Q_local.eval()
            local_q = self.Q_local(states)[np.arange(self.batch_size), actions.numpy()]
            target_actions = self.Q_local(next_states)
            target_q = rewards + self.gamma * target_actions.max(1)[0].detach() * (1 - dones)
            self.Q_local.train()

            ## optimize model
            loss = self.loss(local_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


