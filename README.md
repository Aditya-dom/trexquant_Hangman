# Approaches used
 1. Natural Language Processing (NLP)
 2. Bidirectional Long-Short Term Memory (Bi-LSTM)
 3. Recurrent Neural Network (RNN)
 4. Reinforcement Learning (RL-Deployed Strategy is more than 60% accurate.)
##
1.	Reinforcement Learning: 
I tried my hand at building a reinforcement learning-enabled model to play Hangman. The goal is a win percentage greater than 50%. There are multiple ways to solve this problem, I'd have to imagine, but the problem statement seemed to lend itself well towards RL.

2.	Natural Language Processing (NLP):
Now Drawing inspiration from other Hangman solvers, As I integrated several strategies to approach solving the Hangman problem. It one of them using NLP.
Before any guessing, I update a dictionary of possible words from the training set that match the guessed word-in-progress in terms of length and characters. I calculate the vowel/length ratio of the word-in-progress, and using this information later on, if more than 50% of the letters in the guessed word-in-progress are vowels, then we should not guess a vowel. I also leveraged the nltk library to get the frequencies of n-grams with n from 2 to 5 in the training set.

3.	Bidirectional Long-Short Term Memory
A bidirectional LSTM (BiLSTM) layer is an RNN layer that learns bidirectional long-term dependencies between time steps of time-series or sequence data. These dependencies can be useful when you want the RNN to learn from the complete time series at each time step.

4.	Recurrent Neural Network
Creating an entire Hangman game with an RNN involves multiple components, including data processing, model building, and game logic. Here's an outline of the steps you might take in Python using TensorFlow and Keras.

#
# Idea that I have not used due to Shortage of time:

1.	GRU LAYERS
Unfortunately due to shortage  of time I can't perform the experiment GRU LAYERS.
I can guide you on how to create a flowchart depicting the Hangman game with a GRU (Gated Recurrent Unit) layer.

1. *Start*: Begin with a starting point indicating the initiation of the game.

2. *Initialization*: Initialize the Hangman game by setting up the word to be guessed and the number of attempts allowed.

3. *Input*: Prompt the player to input a character guess.

4. *Check Input*: Verify if the input is a valid character (alphabet) and whether it hasn't been guessed before.

5. *GRU Layer*: Represent the GRU layer where the neural network processes the current game state, including the known letters and their positions in the word.

6. *Prediction*: The GRU layer predicts the most probable next letter or a probability distribution over the alphabet.

7. *Update Game State*: Update the game state based on the prediction. If the predicted letter is correct, reveal its positions in the word; otherwise, decrease the number of attempts remaining.

8. *Check Win/Loss Condition*: Check if the word has been guessed completely or if the attempts have run out.

9. *End*: End the flowchart, indicating either a win or loss. 


## THANK YOU!!




 

## Problem Given by Trexquant Investment lp.
#### ALL RIGHT IS RESERVED TO TREXQUANT INVESTMENT
