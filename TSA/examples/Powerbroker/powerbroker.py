import copy

from tsa.mutation import AppInput, AppInputMutator

class PowerbrokerInput(AppInput):

	def __init__(self, secret):
		super(PowerbrokerInput, self).__init__()
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

class PowerbrokerInputMutator(AppInputMutator):
	pass

class DecreaseBudget(PowerbrokerInputMutator):
	@staticmethod
	def mutate(instance):
		step_size = 10
		result = instance.clone()
		result.bidval = max(result.bidval-step_size, 10)
		return result

class IncreaseBudget(PowerbrokerInputMutator):
	@staticmethod
	def mutate(instance):
		step_size = 10
		result = instance.clone()
		result.bidval = min(result.bidval+step_size, 490)
		return result
