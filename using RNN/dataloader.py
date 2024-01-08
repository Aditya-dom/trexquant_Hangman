import numpy as np
import random
import os
import pickle
import json

np.random.seed(7)

#number of dimensions in input tensor over the vocab size
#1 in this case which represents the blank character
extra_vocab = 1

def filter_and_encode(word, vocab_size, min_len, char_to_id):
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

	encoding = np.zeros((len(word), vocab_size + extra_vocab))
	#dict which stores the location at which characters are present
	#e.g. for 'hello', chars = {'h':[0], 'e':[1], 'l':[2,3], 'o':[4]}
	chars = {k: [] for k in range(vocab_size)}

	for i, c in enumerate(word):
		idx = char_to_id[c]
		#update chars dict
		chars[idx].append(i)
		#one-hot encode
		encoding[i][idx] = 1

	return encoding, [x for x in chars.values() if len(x)], set(list(word))


def batchify_words(batch, vocab_size, using_embedding):
	"""
	converts a list of words into a batch by padding them to a fixed length array
	:param batch: a list of words encoded using filter_and_encode function
	:param: size of vocabulary (26 in our case)
	:param: use_embedding: if True, 
	"""

	total_seq = len(batch)
	if using_embedding:
		#word is a list of indices e.g. 'abd' will be [0,1,3]
		max_len = max([len(x) for x in batch])
		final_batch = []

		for word in batch:
			if max_len != len(word):
				#for index = vocab_size, the embedding gives a 0s vector
				zero_vec = vocab_size*np.ones((max_len - word.shape[0]))
				word = np.concatenate((word, zero_vec), axis=0)
			final_batch.append(word)

		return np.array(final_batch)
	else:
		max_len = max([x.shape[0] for x in batch])
		final_batch = []

		for word in batch:
			#word is a one-hot encoded array of dimensions length x vocab_size
			if max_len != word.shape[0]:
				zero_vec = np.zeros((max_len - word.shape[0], vocab_size + extra_vocab))
				word = np.concatenate((word, zero_vec), axis=0)
			final_batch.append(word)

		return np.array(final_batch)


def encoded_to_string(encoded, target, missed, encoded_len, char_to_id, use_embedding):
	"""
	convert an encoded input-output pair back into a string so that we can observe the input into the model
	encoded: array of dimensions padded_word_length x vocab_size
	target: 1 x vocab_size array with 1s at indices wherever character is present
	missed: 1 x vocav_size array with 1s at indices wherever a character which is NOT in the word, is present
	encoded_len: length of word. Needed to retrieve the original word from the padded word
	char_to_id: dict which maps characters to ids
	use_embedding: if character embeddings are used
	"""

	#get reverse mapping
	id_to_char = {v:k for k, v in char_to_id.items()}

	if use_embedding:
		word = [id_to_char[x] if x < len(char_to_id) - 1 else '*' for x in list(encoded[:encoded_len])]
	else:
		word = [id_to_char[x] if x < len(char_to_id) - 1 else '*' for x in list(np.argmax(encoded[:encoded_len, :], axis=1))]

	word = ''.join(word)
	target = [id_to_char[x] for x in list(np.where(target != 0)[0])]
	missed = [id_to_char[x] for x in list(np.where(missed != 0)[0])]
	print("Word, target and missed characters:", word, target, missed)

#class which constructs database and returns batches during training/testing
class dataloader:

	def __init__(self, mode, config):

		self.mode = mode
		
		self.vocab_size = config['vocab_size']
		#blank vec is the one-hot encoded vector for unknown characters in the word
		self.blank_vec = np.zeros((1, self.vocab_size + extra_vocab))
		self.blank_vec[0, self.vocab_size] = 1
		
		self.batch_size = config['batch_size']
		self.total_epochs = config['epochs']

		#char_to_id is done specifically for letters a-z
		self.char_to_id = {chr(97+x): x for x in range(self.vocab_size)}
		self.char_to_id['BLANK'] = self.vocab_size
		self.id_to_char = {v:k for k, v in self.char_to_id.items()}
		
		self.drop_uniform = config['drop_uniform']
		self.use_embedding = config['use_embedding']

		#dump mapping so that all modules use the same mapping
		if self.mode == 'train':
			with open(config['pickle']+'char_to_id.json', 'w') as f:
				json.dump(self.char_to_id, f)

		#dataset for training and testing
		if mode == 'train':
			filename = config['dataset'] + "250k.txt"
		else:
			filename = config['dataset'] + "20k.txt"

		#if already dumped, load the database from dumped pickle file
		pkl_path = config['pickle'] + mode + '_input_dump.pkl'
		if os.path.exists(pkl_path):
			with open(pkl_path, 'rb') as f:
				self.final_encoded = pickle.load(f)
		else:
			corpus = []
			print("Processing dataset for", self.mode)
			#read .txt file
			with open(filename, 'r') as f:
				corpus = f.readlines()

			self.final_encoded = []
			
			for i, word in enumerate(corpus):
				#print progress
				if i%(len(corpus)//10) == len(corpus)//10-1:
					print("Done:", i+1, "/", len(corpus))
			
				encoding, unique_pos, chars = filter_and_encode(word, self.vocab_size, config['min_len'], self.char_to_id)
				if encoding is not None: #indicates that word length is above threshold
					self.final_encoded.append((encoding, unique_pos, chars))

			#dump encoded database 
			with open(pkl_path, 'wb') as f:
				pickle.dump(self.final_encoded, f)

		#construct input-output pairs
		self.refresh_data(0)

	def refresh_data(self, epoch):
		"""
		constructs a database from the corpus
		each training example consists of 3 main tensors:
		1. encoded word with blanks which represent unknown characters
		2. labels which corresponds to a vector of dimension vocab_size with 1s at indices where characters are to be predicted
		3. miss_chars is a vector of dimension vocab_size with 1s at indcies which indicate that the character is NOT present
		   this information is gained from feedback received from the game i.e. if we predict 'a' and the game returns that 'a' is not present, we update this vector
		   and aske the model to predict again
		"""

		print("Refreshing data")

		#the probability with which we drop characters. As training progresses, the probability increases and
		#hence we feed the model with words which have fewer exisitng characters and more blanks -> makes it more challenging to predict
		drop_prob = 1/(1+np.exp(-epoch/self.total_epochs))
		self.cur_epoch_data = []
		all_chars = list(self.char_to_id.keys())
		all_chars.remove('BLANK')
		all_chars = set(all_chars)

		for i, (word, unique_pos, chars) in enumerate(self.final_encoded):
			#word is encoded vector of dimensions depending on whether we are useimg_embedding or not
			#unique pos is a list of lists which indicates positions of the letters e.g. for 'hello', unique_pos = [[0], [1], [2,3], [4]]
			#chars is a list of characters present in the word. We take it's complement (where all_chars is the sample space)
			#missed chars are randomly chosen from this complement set

			#how many characters to drop
			num_to_drop = np.random.binomial(len(unique_pos), drop_prob)
			if num_to_drop == 0: #drop atleast 1 character
				num_to_drop = 1

			#whether character sets are chosen uniformly or with prob. inversely proportional to number of occurences of each character
			if self.drop_uniform:
				to_drop = np.random.choice(len(unique_pos), num_to_drop, replace=False)
			else:
				prob = [1/len(x) for x in unique_pos]
				prob_norm = [x/sum(prob) for x in prob]
				to_drop = np.random.choice(len(unique_pos), num_to_drop, p=prob_norm, replace=False)

			#positions of characters to drop
			#e.g. word is 'hello', unique_pos = [[0], [1], [2,3], [4]] and to_drop = [[0], [2,3]]
			#then, drop_idx = [0,2,3]
			drop_idx = []
			for char_group in to_drop:
				drop_idx += unique_pos[char_group]
			
			#since we are dropping these characters, it becomes the target for our model
			#note that if a character is repeated, np.sum will give number_of_occurences at that index. We clip it to 1 since loss expects either 0 or 1
			target = np.clip(np.sum(word[drop_idx], axis=0), 0, 1)

			#making sure that a blank character is not a target
			assert(target[self.vocab_size] == 0) 
			target = target[:-1] # drop blank phone
			
			#drop characters and assign blank_character
			input_vec = np.copy(word)
			input_vec[drop_idx] = self.blank_vec

			#if using embedding, we need to provide character id instead of 1-hot encoded vector
			if self.use_embedding:
				input_vec = np.argmax(input_vec, axis=1)
			
			#randomly pick a few characters from vocabulary as characters which were predicted but declared as not present by game
			not_present = np.array(sorted(list(all_chars - chars)))
			num_misses = np.random.randint(0, 10) #10 because most games end before 10 misses
			miss_chars = np.random.choice(not_present, num_misses)
			miss_chars = list(set([self.char_to_id[x] for x in miss_chars]))
			#e.g. word is 'hello', num_misses = 2, miss_chars [1, 3] (which correspond to the characters b and d)
			
			miss_vec = np.zeros((self.vocab_size))
			miss_vec[miss_chars] = 1
			
			#append tuple to list
			self.cur_epoch_data.append((input_vec, target, miss_vec))

		#shuffle dataset before feeding batches to the model
		np.random.shuffle(self.cur_epoch_data)
		self.num_egs = len(self.cur_epoch_data)
		self.idx = 0

	def return_batch(self):
		"""
		returns a batch for trianing/testing
		"""

		cur_batch = self.cur_epoch_data[self.idx: self.idx+self.batch_size]
		#convert to numoy arrays
		lens = np.array([len(x[0]) for x in cur_batch])
		inputs = batchify_words([x[0] for x in cur_batch], self.vocab_size, self.use_embedding)
		labels = np.array([x[1] for x in cur_batch])
		miss_chars = np.array([x[2] for x in cur_batch])

		self.idx += self.batch_size

		if self.idx >= self.num_egs - 1: #indicates end of epoch
			self.idx = 0
			return inputs, labels, miss_chars, lens, 1
		else:
			return inputs, labels, miss_chars, lens, 0

	def __len__(self):

		return len(self.cur_epoch_data)//self.batch_size


if __name__ == '__main__':		

	import yaml
	with open("config.yaml", 'r') as stream:
		try:
			config = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	a = dataloader('test', config)
	c = a.return_batch()
	print(c)
