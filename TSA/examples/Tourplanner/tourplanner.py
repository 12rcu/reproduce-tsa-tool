import random, copy, string

from tsa.mutation import AppInput, AppInputMutator

class TourPlannerInput(AppInput):

	PLACES = (
		"Boston",
		"Worcester",
		"Springfield",
		"Lowell",
		"Cambridge",
		"New Bedford",
		"Brockton",
		"Quincy",
		"Lynn",
		"Fall River",
		"Newton",
		"Lawrence",
		"Somerville",
		"Framingham",
		"Haverhill",
		"Waltham",
		"Malden",
		"Brookline",
		"Plymouth",
		"Medford",
		"Taunton",
		"Chicopee",
		"Weymouth",
		"Revere",
		"Peabody",
	)
	
	@staticmethod
	def abbreviate(placename):
		"""
		Return an abbreviated version of the place name.
		Currently we use the first four nonblank letters,
		which is the minimum needed to avoid any ambiguity.
		"""
		return placename.replace(' ', '')[:4]

	@classmethod
	def checkplacename(cls, placename):
		"""
		Ensure the place name is known, or die.
		"""
		assert placename in cls.PLACES, "Unknown place: " + placename


	def __init__(self, places=None):
		"""
		Constructor given a list of place names. All names must be valid.
		If no list of places is given, a random list of 5 places is chosen.
		"""
		super(TourPlannerInput, self).__init__()
		if places:
			for p in places:
				self.checkplacename(p)
			self.places = places[:]
		else:
			self.places = random.sample(self.PLACES, 5)

	def clone(self):
		"""
		Return a clone of self.
		"""
		return copy.deepcopy(self)

	def getsecret(self):
		"""
		Return the value of the secret for this input.
		The secret is currently the LIST of places to visit (order matters).		
		We return a concatenation of the abbreviated names, e.g., 'Bost_Medf_Taun_Plym_Reve'.
		"""
		return '_'.join(self.abbreviate(p) for p in sorted(self.places))

	def __str__(self):
		"""
		Return plain text representation.
		Currently the full data model is the same as the secret.
		"""
		strs = (
			'places:  {}'.format(self.getsecret()),
		)
		return '\n'.join(strs)

	def replace(self, index, newplacename):
		"""
		Replace the index-th place name with the given one.
		Place name must be valid.
		Place name cannot be already in use elsewhere on the list.
		"""
		self.checkplacename(newplacename)
		assert index < len(self.places), "Cannot replace index " + str(index) + ", only have " + str(len(self.places)) + " elements."
		assert newplacename not in self.places, "New place name is already present: " + newplacename
		self.places[index] = newplacename
	
	def shuffle(self):
		"""
		Shuffle the list of places to visit.
		"""
		random.shuffle(self.places)

	def __hash__(self):
		return hash(tuple(self.places))

class TourPlannerInputMutator(AppInputMutator):
	pass

class Shuffle(TourPlannerInputMutator):
	"""
	Shuffle the whole list of places.
	"""
	@staticmethod
	def mutate(instance):
		#print("Shuffle.mutate()")
		result = instance.clone()
		result.shuffle()
		return result

class ReplaceOne(TourPlannerInputMutator):
	"""
	Replace one of the place names with a random new place name.
	The new name is not used anywhere else on the list, i.e., we
	ensure that there are no duplicates.
	"""
	@staticmethod
	def mutate(instance):
		#print("ReplaceOne.mutate()")
		result = instance.clone()
		used = set(instance.places)
		free = set(instance.PLACES).difference(used)
		index = random.randrange(len(result.places))
		newplace = random.choice(list(free))
		result.replace(index, newplace)
		return result

class ReplaceTwo(TourPlannerInputMutator):
	"""
	Replace two of the place names with a random new place name.
	The new name is not used anywhere else on the list, i.e., we
	ensure that there are no duplicates.
	"""
	@staticmethod
	def mutate(instance):
		#print("ReplaceOne.mutate()")
		result = instance.clone()
		used = set(instance.places)
		free = set(instance.PLACES).difference(used)
		index1 = random.randrange(len(result.places))
		while True:
			index2 = random.randrange(len(result.places))
			if index1 != index2:
				break
		list_free = list(free)
		random.shuffle(list_free)
		newplace1 = list_free[0]
		newplace2 = list_free[1]
		result.replace(index1, newplace1)
		result.replace(index2, newplace2)
		return result

class ReplaceThree(TourPlannerInputMutator):
	"""
	Replace three of the place names with a random new place name.
	The new name is not used anywhere else on the list, i.e., we
	ensure that there are no duplicates.
	"""
	@staticmethod
	def mutate(instance):
		result = instance.clone()
		used = set(instance.places)
		free = set(instance.PLACES).difference(used)
		list_index = range(len(result.places))
		random.shuffle(list_index)
		
		list_free = list(free)
		random.shuffle(list_free)
		
		for i in range(3):
			result.replace(list_index[i], list_free[i])
		return result

class ReplaceFour(TourPlannerInputMutator):
	"""
	Replace four of the place names with a random new place name.
	The new name is not used anywhere else on the list, i.e., we
	ensure that there are no duplicates.
	"""
	@staticmethod
	def mutate(instance):
		result = instance.clone()
		used = set(instance.places)
		free = set(instance.PLACES).difference(used)
		list_index = range(len(result.places))
		random.shuffle(list_index)

		list_free = list(free)
		random.shuffle(list_free)

		for i in range(4):
			result.replace(list_index[i], list_free[i])
		return result

class ReplaceFive(TourPlannerInputMutator):
	"""
	Replace five of the place names with a random new place name.
	The new name is not used anywhere else on the list, i.e., we
	ensure that there are no duplicates.
	"""
	@staticmethod
	def mutate(instance):
		result = instance.clone()
		used = set(instance.places)
		free = set(instance.PLACES).difference(used)
		list_index = range(len(result.places))
		random.shuffle(list_index)

		list_free = list(free)
		random.shuffle(list_free)

		for i in range(5):
			result.replace(i, list_free[i])
		return result
