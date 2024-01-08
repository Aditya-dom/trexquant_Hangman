import gym
from gym import spaces
import string
from gym.spaces.multi_discrete import MultiDiscrete
import numpy as np
from gym.utils import seeding
import random
import collections
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

config = None

MAX_WORDLEN = 25

with open("config.yaml", 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)

logger = logging.getLogger('root')
# logger.warning('is when this event was logged.')


class HangmanEnv(gym.Env):

	def __init__(self):
		# super().__init__()
		self.vocab_size = 26
		self.mistakes_done = 0
		f = open("./words.txt", 'r').readlines()
		self.wordlist = [w.strip() for w in f]
		self.action_space = spaces.Discrete(26)
		self.vectorizer = CountVectorizer(tokenizer=lambda x: list(x))
		# self.wordlist = [w.strip() for w in f]
		self.vectorizer.fit([string.ascii_lowercase])
		self.config = config
		self.char_to_id = {chr(97+x): x for x in range(self.vocab_size)}
		self.char_to_id['_'] = self.vocab_size
		self.id_to_char = {v:k for k, v in self.char_to_id.items()}
		self.observation_space = spaces.Tuple((
			spaces.MultiDiscrete(np.array([25]*27)),     #Current obscured string
			spaces.MultiDiscrete(np.array([1]*26))      #Actions used                      #Wordlen
		))
		self.observation_space.shape=(27, 26)
		self.seed()

	def filter_and_encode(self, word, vocab_size, min_len, char_to_id):
		"""
		checks if word length is greater than threshold and returns one-hot encoded array along with character sets
		:param word: word string
		:param vocab_size: size of vocabulary (26 in this case)
		:param min_len: word with length less than this is not added to the dataset
		:param char_to_id
		"""

		#don't consider words of lengths below a threshold
		word = word.strip().lower()
		if len(word) < min_len:
			return None, None, None

		encoding = np.zeros((len(word), vocab_size + 1))
		#dict which stores the location at which characters are present
		#e.g. for 'hello', chars = {'h':[0], 'e':[1], 'l':[2,3], 'o':[4]}
		chars = {k: [] for k in range(vocab_size+1)}

		for i, c in enumerate(word):
			idx = char_to_id[c]
			#update chars dict
			chars[idx].append(i)
			#one-hot encode
			encoding[i][idx] = 1

		zero_vec = np.zeros((MAX_WORDLEN - encoding.shape[0], vocab_size + 1))
		encoding = np.concatenate((encoding, zero_vec), axis=0)

		return encoding

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def choose_word(self):
		return random.choice(self.wordlist)

	def count_words(self, word):
		lens = [len(w) for w in self.wordlist]
		counter=dict(collections.Counter(lens))
		return counter[len(word)]

	def reset(self):
		self.mistakes_done = 0
		# inputs, labels, miss_chars, input_lens, status = self.dataloader.return_batch()
		self.word = self.choose_word()
		self.wordlen = len(self.word)
		self.gameover = False
		self.win = False
		self.guess_string = "_"*self.wordlen
		self.actions_used = set()
		self.actions_correct = set()
		logger.info("Reset: Resetting for new word")

		logger.info("Reset: Selected word= {0}".format(self.word))


		self.state = (
			self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
			np.array([0]*26)
		)

		logger.debug("Reset: Init State = {self.state}")

		return self.state

	def vec2letter(self, action):
		letters = string.ascii_lowercase
		# idx = np.argmax(action==1)
		return letters[action]

	def getGuessedWord(self, secretWord, lettersGuessed):
		"""
		secretWord: string, the word the user is guessing
		lettersGuessed: list, what letters have been guessed so far
		returns: string, comprised of letters and underscores that represents
		what letters in secretWord have been guessed so far.
		"""
		secretList = []
		secretString = ''
		for letter in secretWord:
			secretList.append(letter)
		for letter in secretList:
			if letter not in lettersGuessed:
				letter = '_'
			secretString += letter
		return secretString


	def check_guess(self, letter):
		if letter in self.word:
			self.prev_string = self.guess_string
			self.actions_correct.add(letter)
			self.guess_string = self.getGuessedWord(self.word, self.actions_correct)
			return True
		else:
			return False

	def step(self, action):
		done = False
		reward = 0
		if string.ascii_lowercase[action.argmax()] in self.actions_used:
			reward = -4.0
			self.mistakes_done += 1
			logger.info("Env Step: repeated action, action was= {0}".format(string.ascii_lowercase[action.argmax()]))
			logger.info("ENV STEP: Mistakes done = {0}".format(self.mistakes_done))
			if self.mistakes_done >= 6:
				done = True
				self.win = True
				self.gameover = True
		elif string.ascii_lowercase[action.argmax()] in self.actions_correct:
			reward = -3.0
			logger.info("Env Step: repeated correct action, action was= {0}".format(string.ascii_lowercase[action.argmax()]))
			logger.info("ENV STEP: Mistakes done = {0}".format(self.mistakes_done))
			# done = True
			# self.win = True
			# self.gameover = True
		elif self.check_guess(self.vec2letter(action.argmax())):
			logger.info("ENV STEP: Correct guess, evaluating reward, guess was = {0}".format(string.ascii_lowercase[action.argmax()]))
			if(set(self.word) == self.actions_correct):
				reward = 10.0
				done = True
				self.win = True
				self.gameover = True
				logger.info("ENV STEP: Won Game, evaluating reward, guess was = {0}".format(string.ascii_lowercase[action.argmax()]))
			# self.evaluate_subset(action)
			reward = +1.0
			# reward = self.edit_distance(self.state, self.prev_string)
			self.actions_correct.add(string.ascii_lowercase[action.argmax()])
		else:
			logger.info("ENV STEP: Incorrect guess, evaluating reward, guess was = {0}".format(string.ascii_lowercase[action.argmax()]))
			self.mistakes_done += 1
			if(self.mistakes_done >= 6):
				reward = -5.0
				done = True
				self.gameover = True
			else:
				reward = -2.0

		self.actions_used.add(string.ascii_lowercase[action.argmax()])

		logger.info("ENV STEP: actions used = {0}".format(" ".join(self.actions_used)))
		logger.info("ENV STEP: actions used = {0}".format(" ".join(self.actions_used)))
		self.state = (
			self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
			self.vectorizer.transform(list(self.actions_used)).toarray()[0]
		)
		logger.debug("Intermediate State = {self.state}")
		return (self.state, reward, done, {'win' :self.win, 'gameover':self.gameover})
