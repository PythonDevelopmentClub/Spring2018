import itertools
import string
import sys

def guess_password(real):
    # chars = string.ascii_lowercase + string.digits
    # key_words = ["Cloud", "atlas"]
    attempts = 0
    key_words = getKeyWords()
    # for password_length in range(8, 14):
    for guess in itertools.product(key_words, repeat=3):
    	for num_length in range(0,10): 
			for num_length in itertools.combinations_with_replacement(string.digits, num_length):
				attempts += 1
		        guess = ''.join(guess)
		        print guess
		        if guess == real:
		            return 'password is {}. found in {} guesses.'.format(guess, attempts)
		        print(guess, attempts)

def getKeyWords(): 
	file = open("keyWords.txt", "r")
	keywords = []
	for line in file:
		keywords.append(line)
	return keywords

def main(): 
	if len(sys.argv) > 1:
		password = sys.argv[1]
	print(guess_password(password))



if __name__=="__main__": 
	main(); 