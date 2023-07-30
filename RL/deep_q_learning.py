import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random

# Agent
class MyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.replay_buffer = deque(maxlen=50000)
        
        self.discount_factor = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.update_target_model = 10

        self.main_network = self.neural_network()
        self.target_network = self.neural_network()
        
        self.target_network.set_weights(self.main_network.get_weights())
        
    def neural_network(self):
        model = Sequential()
        
        model.add(Dense(32, activation='relu', input_shape=(self.state_size, )))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size))
        
        model.compile(optimizer=Adam(self.learning_rate), loss='mse')
        
        return model
    
    def save_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def get_batch(self, batch_size):
        samples = random.sample(self.replay_buffer, batch_size)
        
        state_batch = np.array([batch[0] for batch in samples]).reshape(batch_size, self.state_size)
        action_batch = np.array([batch[1] for batch in samples]).reshape(batch_size)
        reward_batch = np.array([batch[2] for batch in samples]).reshape(batch_size)
        next_state_batch = np.array([batch[3] for batch in samples]).reshape(batch_size, self.state_size)
        done_batch = np.array([batch[4] for batch in samples]).reshape(batch_size)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def train(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.get_batch(batch_size)
        
        q_values = self.main_network.predict(state_batch, verbose=0)
        
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q_values = np.amax(next_q_values, axis=1)
        
        for i in range(batch_size):
            q_values_label = reward_batch[i] if done_batch[i] else reward_batch[i] + self.discount_factor * max_next_q_values[i]
            q_values[i][action_batch[i]] = q_values_label
            
        self.main_network.fit(state_batch, q_values, verbose=0)
            
    
    def make_decision(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state = np.expand_dims(state, axis=0)
        q_values = self.main_network.predict(state, verbose=0)
        return(np.argmax(q_values[0]))
        
        
# Main
env = gym.make("CartPole-v1")
env.reset()
state = env.state

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

n_epochs = 100
n_timesteps = 500
batch_size = 32

my_agent = MyAgent(state_size, action_size)

for i in range(n_epochs):
    env.reset()
    state = env.state
    ep_reward = 0
    
    for timestep in range(n_timesteps):
        if timestep % my_agent.update_target_model == 0:
            my_agent.target_network.set_weights(my_agent.main_network.get_weights())
            
        action = my_agent.make_decision(state)
        
        return_state, reward, terminated, truncated, _ = env.step(action=action)
        done = terminated or truncated
        
        my_agent.save_experience(state, action, reward, return_state, done)
        state = return_state
        
        ep_reward += reward
        
        if done:
            print("Epoch {} reach terminal with reward = {}".format(i + 1, ep_reward))
            break
    
        if (len(my_agent.replay_buffer) > batch_size):
            my_agent.train(batch_size)
            
    if (my_agent.epsilon > my_agent.epsilon_min):
        my_agent.epsilon = my_agent.epsilon * my_agent.epsilon_decay
        
# Save
my_agent.main_network.save("cart_pole_v1.h5")