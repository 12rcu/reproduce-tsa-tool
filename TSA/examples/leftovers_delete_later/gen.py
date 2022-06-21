import random, string

def random_english_words(howmany, minlen=None, maxlen=None):
	WORDS = open('/usr/share/dict/words', 'r').read().splitlines()
	minlen = minlen if minlen else -1    # -inf
	maxlen = maxlen if maxlen else 16384 # +inf
	WORDS = filter(lambda w: len(w) >= minlen and len(w) <= maxlen and set(w).issubset(string.ascii_letters), WORDS)
	return random.sample(WORDS, howmany)

def random_phrase(nwords):
	return '_'.join(random_english_words(nwords, minlen=3, maxlen=7))

def random_integer_digits(ndigits):
	return ''.join([random.choice(string.digits) for i in range(ndigits)])


