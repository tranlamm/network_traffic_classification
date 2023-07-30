import numpy as np
import random
import gym

# Create environment
env = gym.make("MountainCar-v0")

env.reset()

# # GET current state
# print(env.state)

# # Get action space
# print(env.action_space.n)

# # GET state space
# print(env.observation_space.high)
# print(env.observation_space.low)

# while True: 
#     action = 2
#     new_state, reward, done, _, _ = env.step(action)
#     print("New state = {}, Reward = {}, Done = {}".format(new_state, reward, done))
#     env.render()

q_table_size = [20, 20]
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size

def convert_state(state):
    return tuple(np.asarray((state - env.observation_space.low) // q_table_segment_size, dtype=int))

# Initialize Q Table
q_table = np.random.uniform(low=-1, high=1, size=(q_table_size + [env.action_space.n]))

learning_rate = 0.01
discount_factor = 0.9
epochs = 5000
max_reward = -1e6
max_action_list = []
max_begin_state = None

v_epsilon = 0.9
epsilon_decay_start = 1
epsilon_decay_end = epochs // 2 
v_epsilon_decay = v_epsilon / (epsilon_decay_end - epsilon_decay_start)

for epoch in range(0, epochs):
    print("Epoch = ", epoch)
    
    env.reset()
    done = False
    current_state = env.state
    ep_reward = 0
    ep_action_list = []
    ep_begin_state = current_state

    while not done:
        if random.random() > v_epsilon:
            action = np.argmax(q_table[convert_state(current_state)])
        else:
            action = random.randint(0, env.action_space.n - 1)
        
        return_state, reward, terminated, truncated, _ = env.step(action=action)
        
        ep_reward += reward
        ep_action_list.append(action)
        
        done = terminated or truncated
        
        if (done):
            if return_state[0] >= env.goal_position:
                print("Success in epoch = ", epoch)
                if ep_reward > max_reward:
                    max_reward = ep_reward
                    max_action_list = ep_action_list
                    max_begin_state = ep_begin_state
                    
        else:
            new_state = convert_state(return_state)
            
            q_table[convert_state(current_state)][action] += learning_rate * (reward + discount_factor*np.max(q_table[new_state]) - q_table[convert_state(current_state)][action])
            
            current_state = return_state

    v_epsilon -= v_epsilon_decay
env.close()

# Visualize
env_test = gym.make("MountainCar-v0", render_mode="human")
env_test.reset()
env_test.state = max_begin_state

for action in max_action_list:
    env_test.step(action=action)
    env_test.render()
    
env_test.close()
