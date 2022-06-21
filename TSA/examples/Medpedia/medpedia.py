import random, copy, string
from collections import Counter
from itertools import chain, combinations

from tsa.mutation import AppInput, AppInputMutator
from medpedia_client import MedpediaClient

DEBUG_MUTATORS = False
SERVER_HOST = 'serverNuc'
#SERVER_HOST = 'localhost'
SERVER_PORT = 8443

class MedpediaInput(AppInput):
	
	def __init__(self, article=''):
		super(MedpediaInput, self).__init__()
		c = MedpediaClient(SERVER_HOST, SERVER_PORT)
		allarticlenames = c.get_listall_from_file()
		if len(article) == 0:
			self.article = random.choice(allarticlenames)
		else:
			self.article = article

	def clone(self):
		"""
		Return a clone of self.
		"""
		return copy.deepcopy(self)

	def getsecret(self):
		"""
		Return the value of the secret for this input.
		"""
		return self.article
	
	def __str__(self):
		"""
		Return plain text representation.
		"""
		return self.article

	def __hash__(self):
		return hash(self.article)

class MedpediaInputMutator(AppInputMutator):
	pass

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))


def randomword(minlength, maxlength):
	"""
	Return a random string between minlength and maxlength (inclusive).
	"""
	l = random.randint(minlength, maxlength)
	wordchars = [random.choice(string.ascii_letters) for i in range(l)]
	return ''.join(wordchars)

#
# Current min/max lengths for random strings
#

def randomcarid():
	return randomword(4, 12)

def randomcargodescription():
	return randomword(1, 16)

def randompersonname():
	return randomword(4, 12)

def randomstopname():
	return randomword(4, 12)

def randomstoptime():
	hours = random.randint(0, 23)
	minutes = random.randint(0, 59)
	return int('{}{}'.format(hours, minutes))



class ChangeArticle(MedpediaInputMutator):
	"""
	Change the article to be visited to some other article.
	"""
	
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("ChangeArticle.mutate()")
		
		c = MedpediaClient(SERVER_HOST, SERVER_PORT)
		allarticlenames = c.get_listall_from_file()

		result = instance.clone()
		result.article = random.choice(allarticlenames)
		return result

if __name__ == '__main__':
	print("Testing...\n\n")
	for _ in range(5):
		instance = MedpediaInput()
		mutant = ChangeArticle.mutate(instance)
		print(instance)
		print(mutant)
		print("")





