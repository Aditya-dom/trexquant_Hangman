import gym
from env import HangmanEnv
import time
import random
import numpy as np
import pickle
import string



N_EPISODES = 100000
epsilon = 0.1
alpha = 0.1
gamma = 0.8

q_table = {}

games_won = 0
games_played = 0

try:
    with open('q_table.pickle', 'rb') as handle:
        q_table = pickle.load(handle)
  
except:
    print("Unable to find file")

env = HangmanEnv()

for i_episode in range(N_EPISODES):
    state = env.reset()
    games_played += 1
    print("Starting episode")
    if state not in q_table:
        q_table[state] = np.random.rand(26)
    while(1):
        print(state)
        # action = env.action_space.sample()
        # temp = np.zeroes(26)
        
        if random.uniform(0, 1) < epsilon:
            print("Exploring")
            # temp = [0]*26
            # temp[random.randint(0, 25)] = 1
            action = random.randint(0, 25)
        else:
            print("Taking best action")
            action = np.argmax(q_table[state])
        print("action = ", string.ascii_lowercase[action])
        next_state, reward, done, info = env.step(action)
        
        print("reward = ", reward)
        if next_state not in q_table:
            q_table[next_state] = np.random.rand(26)
        
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        
        print("Old val = ", old_value)
        print("next max = ", next_max)
        
        new_value = (1 - alpha)*old_value + alpha * (reward + gamma*next_max)
        q_table[state][action] = new_value
        
        print("new val = ", new_value)
        # print("next max = ", next_max)
        
        state = next_state
        
        if done:
            print("Episode finished", info)
            if info['win'] == True:
                games_won += 1
            break
        # print(q_table)
        # time.sleep(0.1)
        

env.close()

try:
    q_file = open('q_table.pickle', 'wb')
    pickle.dump(q_table, q_file)
    q_file.close()
  
except:
    print("Something went wrong")
    
    
print("Win %:", games_won/games_played)