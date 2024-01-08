## Drawing inspiration from other Hangman solvers, I integrated several strategies to approach solving the Hangman problem.

Before any guessing, I update a dictionary of possble words from the training set that match the guessed word-in-progress in terms of length and characters. I calculate the vowel/length ratio of the word-in-progress, and using this information later on, if more than 50% of the letters in the guessed word-in-progress are vowels, then we should not guess a vowel. I also leveraged the nltk library to get the frequencies of n-grams with n from 2 to 5 in the training set.

Each time a letter is guessed, I update both the dictionary of valid words and frequencies, frequency of valid n-grams that don't have incorrect letters that were guessed previously, and the vowel/length ratio.

I use four "safety nets" calculate the highest weighted, or most likely candidate letter. In the event that one fails, I move onto the next safety net. At each safety net I make the usual precaution regarding vowel frequency:
1) Across all training words that match the word-in-progress, calculate the most frequent letter and return it.
2) For all substrings in length from 3 to half of the length of the word being guessed, from the matching words in the training set, get the frequencies of each letter, and choose the most frequent out of all.
3) Find the largest valid n-grams in length from 2-5 surrounding each '_' index in the word-in-progress, and the most likely letter by frequency, giving priority to longer n-grams. Across all indices, choose the one with the highest frequency.
4) Return to the original training set, calculate frequencies, and return the most likely candidate letter.
