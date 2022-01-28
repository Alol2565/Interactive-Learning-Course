import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from network import DeepNetwork
from experience_replay import ReplayBuffer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent():
    def __init__(self, state_size, action_size, buffer_size = 1e5):
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3
        self.update_freq = 4     
        self.alpha = 5e-4        
        self.buffer_size = int(buffer_size)

        self.state_size = state_size
        self.action_size = action_size

        self.qnetwork_local = DeepNetwork(state_size, action_size).to(device)
        self.qnetwork_target = DeepNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.alpha)

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_freq

        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau )                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)