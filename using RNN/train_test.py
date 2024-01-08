"""
The main driver file responsible for training, testing and predicting
"""

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pickle
from model import RNN
from dataloader import dataloader
from dataloader import encoded_to_string

#load config file
with open("config.yaml", 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)

#class responsible for training, testing and inference
class dl_model():

	def __init__(self, mode):

		# Read config fielewhich contains parameters
		self.config = config
		self.mode = mode

		# Architecture name decides prefix for storing models and plots
		feature_dim = self.config['vocab_size']
		self.arch_name = '_'.join(
			[self.config['rnn'], str(self.config['num_layers']), str(self.config['hidden_dim']), str(feature_dim)])

		print("Architecture:", self.arch_name)
		# Change paths for storing models
		self.config['models'] = self.config['models'].split('/')[0] + '_' + self.arch_name + '/'
		self.config['plots'] = self.config['plots'].split('/')[0] + '_' + self.arch_name + '/'

		# Make folders if DNE
		if not os.path.exists(self.config['models']):
			os.mkdir(self.config['models'])
		if not os.path.exists(self.config['plots']):
			os.mkdir(self.config['plots'])
		if not os.path.exists(self.config['pickle']):
			os.mkdir(self.config['pickle'])

		self.cuda = (self.config['cuda'] and torch.cuda.is_available())

		# load/initialise metrics to be stored and load model
		if mode == 'train' or mode == 'test':

			self.plots_dir = self.config['plots']
			# store hyperparameters
			self.total_epochs = self.config['epochs']
			self.test_every = self.config['test_every_epoch']
			self.test_per = self.config['test_per_epoch']
			self.print_per = self.config['print_per_epoch']
			self.save_every = self.config['save_every']
			self.plot_every = self.config['plot_every']

			# dataloader which returns batches of data
			self.train_loader = dataloader('train', self.config)
			self.test_loader = dataloader('test', self.config)
			#declare model
			self.model = RNN(self.config)

			self.start_epoch = 1
			self.edit_dist = []
			self.train_losses, self.test_losses = [], []

		else:

			self.model = RNN(self.config)

		if self.cuda:
			self.model.cuda()

		# resume training from some stored model
		if self.mode == 'train' and self.config['resume']:
			self.start_epoch, self.train_losses, self.test_losses = self.model.load_model(mode, self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)
			self.start_epoch += 1

		# load best model for testing/inference
		elif self.mode == 'test' or mode == 'test_one':
			self.model.load_model(mode, self.config['rnn'], self.model.num_layers, self.model.hidden_dim)

		#whether using embeddings
		if self.config['use_embedding']:
			self.use_embedding = True
		else:
			self.use_embedding = False

	# Train the model
	def train(self):

		print("Starting training at t =", datetime.datetime.now())
		print('Batches per epoch:', len(self.train_loader))
		self.model.train()

		# when to print losses during the epoch
		print_range = list(np.linspace(0, len(self.train_loader), self.print_per + 2, dtype=np.uint32)[1:-1])
		if self.test_per == 0:
			test_range = []
		else:
			test_range = list(np.linspace(0, len(self.train_loader), self.test_per + 2, dtype=np.uint32)[1:-1])

		for epoch in range(self.start_epoch, self.total_epochs + 1):

			try:

				print("Epoch:", str(epoch))
				epoch_loss = 0.0
				# i used for monitoring batch and printing loss, etc.
				i = 0

				while True:

					i += 1

					# Get batch of inputs, labels, missed_chars and lengths along with status (when to end epoch)
					inputs, labels, miss_chars, input_lens, status = self.train_loader.return_batch()

					if self.use_embedding:
						inputs = torch.from_numpy(inputs).long() #embeddings should be of dtype long
					else:
						inputs = torch.from_numpy(inputs).float()

					#convert to torch tensors
					labels = torch.from_numpy(labels).float()
					miss_chars = torch.from_numpy(miss_chars).float()
					input_lens = torch.from_numpy(input_lens).long()

					if self.cuda:
						inputs = inputs.cuda()
						labels = labels.cuda()
						miss_chars = miss_chars.cuda()
						input_lens = input_lens.cuda()

					# zero the parameter gradients
					self.model.optimizer.zero_grad()
					# forward + backward + optimize
					outputs = self.model(inputs, input_lens, miss_chars)
					loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
					loss.backward()

					# clip gradient
					# torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
					self.model.optimizer.step()

					# store loss
					epoch_loss += loss.item()

					# print loss
					if i in print_range and epoch == 1:
						print('After %i batches, Current Loss = %.7f' % (i, epoch_loss / i))
					elif i in print_range and epoch > 1:
						print('After %i batches, Current Loss = %.7f, Avg. Loss = %.7f, Miss Loss = %.7f' % (
								i, epoch_loss / i, np.mean(np.array([x[0] for x in self.train_losses])), miss_penalty))

					# test model periodically
					if i in test_range:
						self.test(epoch)
						self.model.train()

					# Reached end of dataset
					if status == 1:
						break

				#refresh dataset i.e. generate a new dataset from corpurs
				if epoch % self.config['reset_after'] == 0:
					self.train_loader.refresh_data(epoch)

				#take the last example from the epoch and print the incomplete word, target characters and missed characters
				random_eg = min(np.random.randint(self.train_loader.batch_size), inputs.shape[0]-1)
				encoded_to_string(inputs.cpu().numpy()[random_eg], labels.cpu().numpy()[random_eg], miss_chars.cpu().numpy()[random_eg],
								  input_lens.cpu().numpy()[random_eg], self.train_loader.char_to_id, self.use_embedding)

				# Store tuple of training loss and epoch number
				self.train_losses.append((epoch_loss / len(self.train_loader), epoch))

				# save model
				if epoch % self.save_every == 0:
					self.model.save_model(False, epoch, self.train_losses, self.test_losses,
										  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)

				# test every 5 epochs in the beginning and then every fixed no of epochs specified in config file
				# useful to see how loss stabilises in the beginning
				if epoch % 5 == 0 and epoch < self.test_every:
					self.test(epoch)
					self.model.train()
				elif epoch % self.test_every == 0:
					self.test(epoch)
					self.model.train()
				# plot loss and accuracy
				if epoch % self.plot_every == 0:
					self.plot_loss_acc(epoch)

			except KeyboardInterrupt:
				#save model before exiting
				print("Saving model before quitting")
				self.model.save_model(False, epoch-1, self.train_losses, self.test_losses,
									  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)
				exit(0)


	# test model
	def test(self, epoch=None):

		self.model.eval()

		print("Testing...")
		print('Total batches:', len(self.test_loader))
		test_loss = 0

		#generate a new dataset form corpus
		self.test_loader.refresh_data(epoch)

		with torch.no_grad():

			while True:

				# Get batch of input, labels, missed characters and lengths along with status (when to end epoch)
				inputs, labels, miss_chars, input_lens, status = self.test_loader.return_batch()
				
				if self.use_embedding:
					inputs = torch.from_numpy(inputs).long()
				else:
					inputs = torch.from_numpy(inputs).float()

				labels = torch.from_numpy(labels).float()
				miss_chars = torch.from_numpy(miss_chars).float()
				input_lens= torch.from_numpy(input_lens).long()

				if self.cuda:
					inputs = inputs.cuda()
					labels = labels.cuda()
					miss_chars = miss_chars.cuda()
					input_lens = input_lens.cuda()

				# zero the parameter gradients
				self.model.optimizer.zero_grad()
				# forward + backward + optimize
				outputs = self.model(inputs, input_lens, miss_chars)
				loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
				test_loss += loss.item()

				# Reached end of dataset
				if status == 1:
					break

		#take a random example from the epoch and print the incomplete word, target characters and missed characters
		#min since the last batch may not be of length batch_size
		random_eg = min(np.random.randint(self.train_loader.batch_size), inputs.shape[0]-1)
		encoded_to_string(inputs.cpu().numpy()[random_eg], labels.cpu().numpy()[random_eg], miss_chars.cpu().numpy()[random_eg],
			input_lens.cpu().numpy()[random_eg], self.train_loader.char_to_id, self.use_embedding)

		# Average out the losses and edit distance
		test_loss /= len(self.test_loader)

		print("Test Loss: %.7f, Miss Penalty: %.7f" % (test_loss, miss_penalty))

		# Store in lists for keeping track of model performance
		self.test_losses.append((test_loss, epoch))

		# if testing loss is minimum, store it as the 'best.pth' model, which is used during inference
		# store only when doing train/test together i.e. mode is train
		if test_loss == min([x[0] for x in self.test_losses]) and self.mode == 'train':
			print("Best new model found!")
			self.model.save_model(True, epoch, self.train_losses, self.test_losses,
								  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)

		return test_loss

	def predict(self, string, misses, char_to_id):
		"""
		called during inference
		:param string: word with predicted characters and blanks at remaining places
		:param misses: list of characters which were predicted but game feedback indicated that they are not present
		:param char_to_id: mapping from characters to id
		"""

		id_to_char = {v:k for k,v in char_to_id.items()}

		#convert string into desired input tensor
		if self.use_embedding:
			encoded = np.zeros((len(char_to_id)))
			for i, c in enumerate(string):
				if c == '*':
					encoded[i] = len(id_to_char) - 1 
				else:
					encoded[i] = char_to_id[c]

			inputs = np.array(encoded)[None, :]
			inputs = torch.from_numpy(inputs).long()

		else:

			encoded = np.zeros((len(string), len(char_to_id)))
			for i, c in enumerate(string):
				if c == '*':
					encoded[i][len(id_to_char) - 1] = 1
				else:
					encoded[i][char_to_id[c]] = 1

			inputs = np.array(encoded)[None, :, :]
			inputs = torch.from_numpy(inputs).float()

		#encode the missed characters
		miss_encoded = np.zeros((len(char_to_id) - 1))
		for c in misses:
			miss_encoded[char_to_id[c]] = 1
		miss_encoded = np.array(miss_encoded)[None, :]
		miss_encoded = torch.from_numpy(miss_encoded).float()

		input_lens = np.array([len(string)])
		input_lens= torch.from_numpy(input_lens).long()	

		#pass through model
		output = self.model(inputs, input_lens, miss_encoded).detach().cpu().numpy()[0]

		#sort predictions
		sorted_predictions = np.argsort(output)[::-1]
		
		#we cannnot consider only the argmax since a missed character may also get assigned a high probability
		#in case of a well-trained model, we shouldn't observe this
		return [id_to_char[x] for x in sorted_predictions]

	def plot_loss_acc(self, epoch):
		"""
		take train/test loss and test accuracy input and plot it over time
		:param epoch: to track performance across epochs
		"""

		plt.clf()
		fig, ax1 = plt.subplots()

		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')
		ax1.plot([x[1] for x in self.train_losses], [x[0] for x in self.train_losses], color='r', label='Train Loss')
		ax1.plot([x[1] for x in self.test_losses], [x[0] for x in self.test_losses], color='b', label='Test Loss')
		ax1.tick_params(axis='y')
		ax1.legend(loc='upper left')

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.grid(True)
		plt.legend()
		plt.title(self.arch_name)

		filename = self.plots_dir + 'plot_' + self.arch_name + '_' + str(epoch) + '.png'
		plt.savefig(filename)

		print("Saved plots")


if __name__ == '__main__':

	a = dl_model('train')
	a.train()
	# char_to_id = {chr(97+x): x+1 for x in range(26)}
	# char_to_id['PAD'] = 0
	# a = dl_model('test_one')
	# print(a.predict("*oau", char_to_id))
