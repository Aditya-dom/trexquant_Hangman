import logging

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27

# create logger
logger = logging.getLogger('root')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = None


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()       
        num_classes = 26
        num_layers = 1
        input_size = 27
        hidden_size = 32
        seq_length = 27
        
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size+26, num_classes) #fully connected 1
        # self.softmax = nn.LogSoftmax(dim=2)
        self.fc = nn.Linear(num_classes, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,word, actions):
        logger.debug("Forwad:word shape = {0}".format(word.shape))
        logger.debug("Forward:actions shape = {0}".format(actions.shape))
        logger.debug("Forward:actions = {0}".format(actions))
        logger.debug("Forward:word = {0}".format(word))
        # actions = torch.tensor(actions.reshape(-1, 26))
        # print(h_0.requires_grad)
        # Propagate input through LSTM
        # print("Forward: word req grad", word.requires_grad)
        output, (hn, cn) = self.lstm(word.float()) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        logger.debug("Forward: hn shape = {0}".format(hn.shape))
        combined = torch.cat((hn, actions), 1)
        out = self.relu(combined)
        logger.debug("Forward: combined shape = {0}".format(combined.shape))
        out = self.fc_1(out) #first Dense
        out = self.fc(out)
        logger.debug("Forward: Out = {0}".format(out))
        # out_binary = out.argmax(1)
        # print("out binary = ", out_binary)
        # final_action = torch.zeros(out.shape).scatter(1, out_binary.unsqueeze (1), 1).long()
        # print("Forward: out = ", out.numpy().tolist())
        # out = self.relu(out) #relu
        # out = self.fc(out) #Final Output
        # out = self.softmax(out)
        # print("Forward: out =", out.argmax())
        return out
    