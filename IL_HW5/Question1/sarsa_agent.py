from amalearn.agent import AgentBase
import sys
import numpy as np
from tile_coder import MountainCarTileCoder

def epsilon_greedy_policy(w, epsilon, num_actions):
    def policy_fn(tiles):
        A = np.zeros(num_actions) + 1
        action_values = np.zeros(num_actions)
        for i in range(num_actions):
            action_values[i] = w[i][tiles].sum()
        best_action = np.argmax(action_values)
        A = A * epsilon/len(A)
        A[best_action] += 1 - epsilon
        return A, action_values
    return policy_fn

class Agent_Sarsa(AgentBase):
    def __init__(self, id, environment, discount, alpha,epsilon, decay, iht_size, num_tilings, num_tiles):
        self.env = environment
        self.alpha = alpha
        self.epsilon = epsilon
        self.alpha_decay = decay[0]
        self.epsilon_decay = decay[1]
        self.discount_factor = discount
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht_size = iht_size
        self.num_actions = 3
        
        self.mct = MountainCarTileCoder(iht_size, num_tilings, num_tiles)

        self.initial_weights = np.zeros((self.num_actions, iht_size))
        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights

        super(Agent_Sarsa, self).__init__(id, environment)
        
    def run(self, trail, max_time):
        step_episode = []
        for i_episode in range(1, trail+1):
            state = self.env.reset()
            [position, velocity] = state
            active_tiles = self.mct.get_tiles(position, velocity)
            behavior_policy = epsilon_greedy_policy(self.w, self.epsilon, self.num_actions)
            probs, q_vals = behavior_policy(active_tiles)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            q_val = q_vals[action]
            
            if(self.alpha_decay):
                self.alpha = max(0.9 * self.alpha, 5e-6)

            if(self.epsilon_decay):
                self.epsilon = max(0.95 * self.epsilon, 0.001)

            for t in range(max_time):
                if(i_episode == trail + 1):
                    self.env.render()
                #Take action A, observe R,S'
                next_state, reward, done, _ = self.env.step(action)
                [next_position, next_velocity] = next_state
                next_active_tiles = self.mct.get_tiles(next_position, next_velocity)

                # if S' is terminal
                if done:
                    q_val = self.w[action][active_tiles].sum()
                    delta = reward - q_val
                    grad = np.zeros_like(self.w)
                    grad[action][active_tiles] = 1
                    self.w += self.alpha * delta * grad
                    step_episode.append(t)
                    print("\rEpisode {}/{} | steps: {} | Epsilon: {:.5f} | Alpha : {:.5f}".format(i_episode, trail, t,self.epsilon, self.alpha ), end="")
                    sys.stdout.flush()
                    break

                #Choose A' as a function of q(s',.w)
                behavior_policy = epsilon_greedy_policy(self.w, self.epsilon, self.num_actions)
                next_action_probs, next_q_vals = behavior_policy(next_active_tiles)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                next_q_val = next_q_vals[next_action]
                
    
                q_val = self.w[action][active_tiles].sum()
                delta = reward + self.discount_factor * next_q_val - q_val
                grad = np.zeros_like(self.w)
                grad[action][active_tiles] = 1
                self.w += self.alpha * delta * grad
                state = next_state
                [position, velocity] = state
                active_tiles = self.mct.get_tiles(position, velocity)
                action = next_action
        return step_episode