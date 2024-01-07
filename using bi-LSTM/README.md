# trexquantHangmanChallenge
Hangman challenge by trexquant is a challenge to predict words, letter by letter.

## Vowel Prior Probability
Vowels: [a, e, i, o, u]
Created dictionary vowel_prior such that
vowel_prior = {}
keys: length of words
values: the probability of vowels given the length of words

## Data Encoding
### Input Data
Permutation:
From ~220k words dictionary, we have created around 10 million words by masking different letters in the word, i.e., by replacing letters with underscore.

The maximum length of a word in the given dictionary is 29. Testing will happen on mutually exclusive datasets. Thus max word length is assumed at 35.
Each input word is encoded to a 35-dimensional vector, such that alphabets {a-z} will be replaced by numbers {1-26} and underscore will be replaced by 27. The vector will be pre-padded.
Thus, the masked word "aa__" of the word "aaa" will be encoded as 
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,27]

### Target Data
For each of the masked input words, the output will be the original unmasked word. This word has been encoded into a 26-dimensional vector with each position representing letters of the alphabet from a to z.
Thus the output encoding for the word "aaa" will be:
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

As the word contains only one letter "a", output encoding will have 1 at the first position.

## Modelling
Encoding + bi-LSTM has been built to train on this data.

## Prediction Strategy
It is required to predict the word within 6 incorrect tries.

1. Vowel Prediction:
   Leveraging Vowel_priors, we will guess the top vowel if
     tries_remains > 4 and len(guessed_letters) <= max_vowel_guess_limit
2. The remaining tries will be utilized by the bi-lstm model
3. The prediction will happen letter-by-letter
