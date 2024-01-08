import numpy as np
import json
from train_test import dl_model
import matplotlib.pyplot as plt

#class responsible for actual gameplay
class Game:

    def __init__(self, char_to_id_pth):

        #load character to id mapping dumped by dataloader
        with open(char_to_id_pth, 'r') as f:
            self.char_to_id = json.load(f)

        #initialise model
        self.dl_model = dl_model('test_one')

    def play(self):

        #store correctly and incorresclty predicted characters
        misses, hits = [], []
        print("Instructions: For every predicted letter,")
        print("1. Type 'x' if letter does not exists in your word.")
        print("2. If it does exist, input the places at which it exists.")
        print("For e.g., if the word is hello and model's prediction is 'l', type 3 4 in the prompt. If prediction is 'a', type x.\n")

        print("Think of a word and input the number of characters:")
        num_chars = int(input())
        #initialise target string with blanks
        predicted_string = ['*']*num_chars
        
        while 1:

            #get sorted predictions according to probability
            best_chars = self.dl_model.predict(predicted_string, misses, self.char_to_id)
            #best char is the one with highest probability AND which does not belong to list of incorrectly predicted chars
            #AND not already present in the target string
            for pred in best_chars:
                if pred not in misses and pred not in predicted_string:
                    best_char = pred
                    break
            
            #predict and ask user for feedback
            print("Prediction: " + best_char + "\nWhat do ya think?")

            while 1:
                #get input
                inp = input().strip()
                if inp == 'x': #denotes character not present
                    output = 0
                    break
                try:
                    #if it is present, user returns a list of indices at which character is present (note that indexing begins from 1 for user feedback)
                    output = [int(x) for x in inp.split(' ')]
                    #update target string
                    for pred in output:
                        predicted_string[pred-1] = best_char
                    break
                except:
                    print("Invalid format. Please refer to instructions.") 
                    continue


            if output == 0: #indicates miss
                print("Miss")
                #append to missed characters list
                misses.append(best_char)
            else:
                #correctly predicted
                if '*' in predicted_string: #indicates game is not yet over since we still have unknown characters in target string
                    print("Guess correct! New target: " + ''.join(predicted_string))
                    hits.append(best_char)
                else: #indicates game is over. Report number of misses and return
                    print("Game over. Predicted word: " + ''.join(predicted_string))
                    print("Total misses: ", len(misses))
                    return misses

    def test_performance(self, dataset_pth='dataset/20k.txt', num_trials=100, min_word_len=3, plot=True):

        with open(dataset_pth, 'r') as f:
            words = f.readlines()

        words = [x.strip() for x in words if len(x) >= min_word_len + 1] #+1 since /n is yet to be stripped
        #randomly choose words from corpus
        to_test = np.random.choice(words, num_trials)
        print("Testing performance on the following words:", to_test)

        #stores information about average misses for various lengths of target words
        len_misses_dict = {}
        
        for word in to_test:
            #intialise dict
            if len(word) not in len_misses_dict:
                len_misses_dict[len(word)] = {'misses': 0, 'num': 0}

            hits, misses = [], []
            predicted_string = ['*']*len(word)

            #keep predicting
            while 1:
                #get sorted predictions according to probability
                best_chars = self.dl_model.predict(predicted_string, misses, self.char_to_id)
                #best char is the one with highest probability AND which does not belong to list of incorrectly predicted chars
                #AND not already present in the target string
                for pred in best_chars:
                    if pred not in misses and pred not in predicted_string:
                        best_char = pred
                        break

                found_char = False
                if best_char in word: #denotes character not present
                    #if it is present, user returns a list of indices at which character is present (note that indexing begins from 1 for user feedback)
                    indices = []
                    for i, c in enumerate(word):
                        if c == best_char:
                            indices.append(i)
                    #update target string
                    for pred in indices:
                        predicted_string[pred] = best_char
                    found_char = True

                if found_char is False: #indicates miss
                    #append to missed characters list
                    misses.append(best_char)
                else:
                    #correctly predicted
                    if '*' in predicted_string: #indicates game is not yet over since we still have unknown characters in target string
                        hits.append(best_char)
                    else: #indicates game is over. Report number of misses and return
                        len_misses_dict[len(word)]['misses'] += len(misses)
                        len_misses_dict[len(word)]['num'] += 1
                        break

        len_misses_list = [(l, x['misses']/x['num']) for l, x in len_misses_dict.items()]
        len_misses_list = sorted(len_misses_list, key = lambda x: x[0])
        print("Average number of misses:", len_misses_list)

        #plot performance
        if plot:
            plt.bar([x[0] for x in len_misses_list], [x[1] for x in len_misses_list])
            plt.xlabel('Length of word')
            plt.ylabel('Average misses (lesser the better)')
            plt.title("Comparing performance as a function of word length")
            plt.xticks(list(range(min_word_len, len_misses_list[-1][0])))
            plt.show()


a = Game('pickle/char_to_id.json')
# a.play()
a.test_performance()