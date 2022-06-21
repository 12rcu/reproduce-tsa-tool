import copy

class AppInput(object):
	"""
	Abstract base class: one input for an application.
	(To be subclassed for each application.)
	"""

	def __init__(self) -> None:
		"""
		Create an AppInput. Nothing here for now.
		"""
		pass

	def clone(self):
		"""
		Return a clone of self.
		"""
		return copy.deepcopy(self)
		
	def getsecret(self):
		"""
		Return the value of the secret for this input.
		"""
		raise NotImplementedError

	def __str__(self) -> str:
		"""
		Return a plain text representation of this input.
		"""
		raise NotImplementedError
	
	def __hash__(self) -> int:
		"""
		Return a hash value for this input.
		"""
		pass

class AppInputMutator(object):
	"""
	Abstract base class for a mutator object that takes a valid instance
	of a certain type of AppInput and returns another valid instance.
	
	The definition of 'valid' is not obvious. In this context we mean that the input
	instance complies with any representation invariants hold, and when run through
	the application, will not crash or break the application's state, so that it
	remains possible to run multiple inputs one after another.
	"""
	@staticmethod
	def mutate(input):
		"""
		Return a mutated copy of input, or None if impossible to mutate
		(i.e., if this kind of mutation cannot be applied to this input
		without breaking some invariant).

		Assumes that the provided input complies with the representation invariants
		and that it would not crash/break the application state.

		Guarantees that those nice properties will be preserved in the returned
		mutated input (unless we return None due to mutation being infeasible).
		"""
		raise NotImplementedError
