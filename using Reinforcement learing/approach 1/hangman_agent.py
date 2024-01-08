from re import T
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from torch.cuda import init
from env import HangmanEnv
from torch.autograd import Variable
from config import Config
import yaml
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from memory import Transition, ReplayMemory
from dqn import DQN
from log import setup_custom_logger
import time
# import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27

# create logger
logger = setup_custom_logger('root', "./latest.log", "INFO")
logger.propagate = False
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = None
        
class HangmanPlayer():
    def __init__(self, env, config):
        self.memory = None
        self.steps_done = 0
        self.episode_durations = []
        self.last_episode = 0
        self.reward_in_episode = []
        self.env = env
        self.id = int(time.time())
        self.config = config
        self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compile()
        
    def compile(self):     
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # summary(self.target_net, (128, 25, 27))
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
    def _update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def _adjust_learning_rate(self, episode):
        delta = self.config.training.learning_rate - self.config.optimizer.lr_min
        base = self.config.optimizer.lr_min
        rate = self.config.optimizer.lr_decay
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
    def _get_action_for_state(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("This value is ", policy_net(state).max(1))
                selected_action = self.policy_net(torch.tensor(state[0]), torch.tensor([state[1]])).argmax()
                # print("ActionSelect: require grad = ", tens.requires_grad)
                value = selected_action.numpy()
                b = np.zeros((value.size, 26))
                # print("Value = ", value)
                b[np.arange(value.size), value] = 1
                selected_action = torch.from_numpy(b).long()
                # print("from net, got action = ", int(value))
                # final_action = torch.zeros(26).scatter(1, selected_action.unsqueeze (1), 1).long()
                # print("Final action2 = ", selected_action)
                return selected_action
        else:
            a = np.array(random.randrange(self.n_actions))
            b = np.zeros((1, 26))
            b[0, a] = 1
            print(b.shape)
            selected_action = torch.from_numpy(b).long()
            # print("ActionSelect: action selected = ", type(random.randrange(self.n_actions)))
            # final_action = torch.zeros(26).scatter(1, selected_action.unsqueeze (1), 1).long()
            # print("Final action = ", selected_action)
            return selected_action
        
    def save(self):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "reward_in_episode": self.reward_in_episode,
            "episode_durations": self.episode_durations,
            "config": self.config
            }, f"./models/pytorch_{self.id}.pt")
        
    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(torch.tensor([state], device=self.device),
                        torch.tensor([action], device=self.device, dtype=torch.long),
                        torch.tensor([next_state], device=self.device),
                        torch.tensor([reward], device=self.device),
                        torch.tensor([done], device=self.device, dtype=torch.bool))
    
    def fit(self):
        num_episodes = 5000000
        self.memory = ReplayMemory(10000)
        self.episode_durations = []
        self.reward_in_episode = []
        reward_in_episode = 0
        self.epsilon_vec = []
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            # self.check_grad()
            # last_screen = get_screen()
            # current_screen = get_screen()
            # print("Fit: state = ", state)
            state = (state[0].reshape(-1, 25, 27), state[1])
            # state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                # self.check_grad()
                action = self._get_action_for_state(state)
                # self.check_grad()
                # print("Fit: action = ", action.shape)
                next_state, reward, done, info = self.env.step(action)
                
                # reward = torch.tensor([reward], device=device)
                next_state = (next_state[0].reshape(-1, 25, 27), next_state[1])        # Observe new state
                # action_vector = next_state[1]
                # print("Fit: next state actions = ", next_state[1])
                # last_state = state
                # state=next_state
                # current_screen = get_screen()
                # if not done:
                #     next_state = current_screen - last_screen
                # else:
                #     next_state = None

                # Store the transition in memory
                self._remember(state[0], next_state[1], next_state[0], reward, done)
                
                if i_episode >= self.config.training.warmup_episode:
                    self._train_model()
                    self._adjust_learning_rate(i_episode - self.config.training.warmup_episode + 1)
                    done = (t == self.config.rl.max_steps_per_episode - 1) or done
                else:
                    done = (t == 5 * self.config.rl.max_steps_per_episode - 1) or done
                
                # Move to the next state
                state = next_state
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(t + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    # self.epsilon_vec.append(epsilon)
                    reward_in_episode = 0
                    break

                # Update the target network, copying all weights and biases in DQN
                if i_episode % 50 == 0:
                    self._update_target()

                # if i_episode % self.config.training.save_freq == 0:
                #     self.save()

                self.last_episode = i_episode
                # Move to the next state

                # Perform one step of the optimization (on the policy network)
                if done:
                    self.episode_durations.append(t + 1)
                    # plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.config.rl.target_model_update_episodes == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if i_episode % self.config.training.save_freq == 0:
                self.save()
    
    def _train_model(self):  
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_batch.resize_(BATCH_SIZE, 25, 27)
        non_final_next_states.resize_(BATCH_SIZE, 25, 27)
        # print("action batch = ", action_batch.numpy().sum())
        # action_batch.resize_(BATCH_SIZE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print("Here printing = ", self.policy_net(state_batch, action_batch))
        state_action_values = self.policy_net(state_batch, action_batch).gather(0, action_batch)
        # print("State batch = ", state_batch)
        # print("action batch = ", action_batch.shape)
        # print("reward batch = ", reward_batch.shape)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" self.target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float)
        temp = self.target_net(non_final_next_states, action_batch).max(1)[0].detach()
        # print("TrainModel: temp = ", temp.numpy().tolist())
        # print(next_state_values)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, action_batch).max(1)[0].detach()
        # Compute the expected Q values
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # print("TrainModel: expected state action values = ", expected_state_action_values)
        # print("TrainModel: state action values = ", state_action_values.numpy().tolist())
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # clone = expected_state_action_values.clone()
        # with torch.enable_grad():
        # print(next_state_values.shape)
        # print(state_action_values.shape)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)).float()
        # print(expected_state_action_values.requires_grad)
        logger.info("trainmodel: loss = {0}".format(loss))
        # loss.requires_grad = True
        self.optimizer.zero_grad()
        # loss = Variable(loss, requires_grad = True)
        # Optimize the model
        # loss.retain_grad()
        # print(loss.grad)
        loss.backward()

        # print("TrainModel: params = ", list(self.policy_net.parameters()))
        for name, param in self.policy_net.named_parameters():
            # print("Param = ", name, param.is_leaf)
            # try:
            # param.is_leaf
            param.grad.data.clamp_(-1, 1)
            # break
            # except:
            #     print("Failed")
            #     # pass
        self.optimizer.step()
        
    def play(self, verbose:bool=False, sleep:float=0.2, max_steps:int=100):
        # Play an episode
        try:
            actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

            iteration = 0
            state = self.env.reset()  # reset environment to a new, random state
            state = (state[0].reshape(-1, 25, 27), state[1])
            if verbose:
                print(f"Iter: {iteration} - Action: *** - Reward ***")
            time.sleep(sleep)
            done = False

            while not done:
                action = self._get_action_for_state(state)
                iteration += 1
                state, reward, done, info = self.env.step(action)
                display.clear_output(wait=True)
                self.env.render()
                if verbose:
                    print(f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}")
                time.sleep(sleep)
                if iteration == max_steps:
                    print("cannot converge :(")
                    break
        except KeyboardInterrupt:
            pass
            
    def evaluate(self, max_steps:int=100):
        try:
            total_steps, total_penalties = 0, 0
            episodes = 100

            for episode in trange(episodes):
                state = self.env.reset()  # reset environment to a new, random state
                state = (state[0].reshape(-1, 25, 27), state[1])
                nb_steps, penalties, reward = 0, 0, 0

                done = False

                while not done:
                    action = self._get_action_for_state(state)
                    state, reward, done, info = self.env.step(action)

                    if reward == -10:
                        penalties += 1

                    nb_steps += 1
                    if nb_steps == max_steps:
                        done = True

                total_penalties += penalties
                total_steps += nb_steps

            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps / episodes}")
            print(f"Average penalties per episode: {total_penalties / episodes}")    
        except KeyboardInterrupt:
            pass