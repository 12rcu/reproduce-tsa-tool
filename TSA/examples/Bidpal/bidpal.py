import copy
from itertools import chain, combinations

from tsa.mutation import AppInput, AppInputMutator

class BidpalInput(AppInput):

	def __init__(self, secret):
		super(BidpalInput, self).__init__()
		#Our input is a list of 1s and 0s
		self.bidval = secret

	def clone(self):
		""" 
		Return a clone of self.
		"""
		return copy.deepcopy(self)

	def getsecret(self):
		""" 
		Return the value of the secret for this input.
		"""
		return str(self.bidval)

	def __str__(self):
		""" 
		Return plain text representation.
		"""
		return 'secret:{}'.format(self.bidval)

	def __hash__(self):
		return self.bidval

class BidpalInputMutator(AppInputMutator):
	pass

def powerset(iterable):
	""" 
	powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
	"""
	xs = list(iterable)
	# note we return an iterator rather than a list
	return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

class DecreaseBid(BidpalInputMutator):
	@staticmethod
	def mutate(instance):
		step_size = 10
		result = instance.clone()
		result.bidval = max(result.bidval-step_size, 10)
		return result

class IncreaseBid(BidpalInputMutator):
	@staticmethod
	def mutate(instance):
		step_size = 10
		result = instance.clone()
		result.bidval = min(result.bidval+step_size, 490)
		return result




