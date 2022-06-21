import random, copy

from tsa.mutation import AppInput, AppInputMutator

DEBUG_MUTATORS = False

class GabfeedInput(AppInput):

	def __init__(self, secret):
		super(GabfeedInput, self).__init__()
		#Our input is a list of 1s and 0s
		self.auth_key = [1]*secret + [0]*(64-secret)
		random.shuffle(self.auth_key)

	def clone(self):
		"""
		Return a clone of self.
		"""
		return copy.deepcopy(self)

	def getsecret(self):
		"""
		Return the value of the secret for this input.
		"""
		return str(sum(self.auth_key))

	def __str__(self):
		"""
		Return plain text representation.
		"""
		return ''.join([str(x) for x in self.auth_key])

	def getinput(self):
		return ''.join([str(x) for x in self.auth_key])
	
	def __hash__(self):
		return int(self.getinput())

class GabfeedInputMutator(AppInputMutator):
	pass

class AddOnes(GabfeedInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("AddCar.mutate()")
		old_secret = int(instance.getsecret())
		if old_secret >= 64:
			return None
		#result = instance.clone()
		#TODO Do a more proper adding ones instead of generating a new one from scratch.
		result = GabfeedInput(min(old_secret+1, 64))

		return result

class RemoveOnes(GabfeedInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("RemoveCar.mutate()")
		old_secret = int(instance.getsecret())
		if old_secret <= 0:
			return None

		#result = instance.clone()
		result = GabfeedInput(max(old_secret-1, 0))
		return result

class AddFives(GabfeedInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		old_secret = int(instance.getsecret())
		if old_secret >= 64:
			return None
		result = GabfeedInput(min(old_secret+5, 64))

		return result

class RemoveFives(GabfeedInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		old_secret = int(instance.getsecret())
		if old_secret <= 0:
			return None

		result = GabfeedInput(max(old_secret-5, 0))
		return result

class ShuffleKey(GabfeedInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("ShuffleKey.mutate()")
		result = instance.clone()
		random.shuffle(result.auth_key)
		return result




