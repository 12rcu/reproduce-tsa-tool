import random, copy, string
from collections import Counter
from itertools import chain, combinations

from tsa.mutation import AppInput, AppInputMutator


# We only model platform A for now. This simplifies stateless execution.

# We only model single-unit packs of cargo, which avoids having to deal
# with the complexity of reimplementing all the packing algorithms just
# to know what the secret is.


DEBUG_MUTATORS = False


class RailyardInput(AppInput):

	MATERIALS = ('paper', 'lumber', 'crates', 'coal', 'livestock', 'passengers')
	
	CAR_TYPES = {
		'Box Car 1':      ('crates', 5),
		'Box Car 2':      ('crates', 6),
		'Caboose':        ('passengers', 1),
		'Coal Car':       ('coal', 2),
		'Flatbed Car':    ('lumber', 10),
		'Livestock Car':  ('livestock', 6),
		'Mail Car':       ('paper', 25),
		'Passenger Car':  ('passengers', 8),
	}
	
	#MATERIAL_TO_CAR_TYPES = dict([(y, x) for (x, (y,_)) in CAR_TYPES.items()])
	MATERIAL_TO_CAR_TYPES = {
		'crates':         set(['Box Car 1', 'Box Car 2']),
		'passengers':     set(['Caboose', 'Passenger Car']),
		'livestock':      set(['Livestock Car']),
		'paper':          set(['Mail Car']),
		'coal':           set(['Coal Car']),
		'lumber':         set(['Flatbed Car']),
	}
	
	def __init__(self):
		super(RailyardInput, self).__init__()
		# We model cars with names but no order.
		self.cars = dict((ct, set()) for ct in self.CAR_TYPES)
		# We model the cargo with names but no order (in the real app, order matters).
		self.cargo = dict((m, set()) for m in self.MATERIALS)
		# We model personnel as a set of names (in the real app, order matters).
		self.personnel = set()
		# We store the schedule as a dict of (int, str) tuples.
		# We order the schedule entries canonically (by int) before any usage.
		# We will have no duplicate ints and no duplicate strs.
		self.schedule = dict()

	def clone(self):
		"""
		Return a clone of self.
		"""
		return copy.deepcopy(self)

	def getsecret(self):
		"""
		Return the value of the secret for this input.
		"""
		# Get the first 3 letters of each cargo of which there are >0 units.
		symbols = [m[:3] for m in self.MATERIALS if len(self.cargo.get(m)) > 0]
		# Return a canonical textual representation (sorted & concatenated).
		if symbols:
			return '_'.join(sorted(symbols))
		else:
			return 'noc' # no cargo!
	
	def __str__(self):
		"""
		Return plain text representation.
		"""
		strs = (
			#'cars:      {}'.format(list(sorted(self.cars.elements()))),
			'cars:      {}'.format(self.cars),
			'cargo:     {}'.format(self.cargo),
			'personnel: {}'.format(self.personnel),
			'schedule:  {}'.format(self.schedule),
			'secret:    {}'.format(self.getsecret()),
		)
		return '\n'.join(strs)
	
	def gettotalcapacity(self):
		"""
		Return a multiset (Counter) with the total capacity per material.
		This is the sum of capacities of all cars currently in the train.
		"""
		result = Counter(dict((m, 0) for m in self.MATERIALS))
		for car_type, names_of_cars_of_this_type in self.cars.iteritems():
			number_of_cars_of_this_type = len(names_of_cars_of_this_type)
			for i in range(number_of_cars_of_this_type):
				material, car_capacity = self.CAR_TYPES[car_type]
				result[material] += car_capacity
		return result
	
	def getusedcapacity(self):
		"""
		Return a multiset (Counter) with the used capacity per material.
		This is the cargo currently on the platform.
		"""
		return Counter(dict([(material, len(names)) for (material, names) in self.cargo.iteritems()]))
	
	def getfreecapacity(self):
		"""
		Return a multiset (Counter) with the remaining capacity per material.
		This is the total capacities minus the currently used capacities.
		"""
		return self.gettotalcapacity() - self.getusedcapacity()

	def getnumberofcars(self):
		"""
		Return the total number of cars.
		"""
		return sum(len(names) for (cartype, names) in self.cars.iteritems())

	def addcar(self, car_type, car_id):
		"""
		Add a car of the given type with the given name.
		There can be no other car of any type with that name.
		"""
		assert car_type in self.CAR_TYPES.keys()
		assert not self.caridinuse(car_id), "Car ID already in use: {}".format(car_id)
		self.cars[car_type].add(car_id)

	def caridinuse(self, car_id):
		"""
		Return True if there is some car with this ID.
		"""
		return any((car_id in names) for names in self.cars.itervalues())
		
	def removeonecar(self, car_type):
		"""
		Remove some car of the given type.
		PEND: This should assert that no excess cargo is left behind!
		"""
		assert car_type in self.CAR_TYPES.keys()
		assert len(self.cars[car_type]) > 0
		name = random.choice(list(self.cars[car_type]))
		self.cars[car_type].remove(name)

	def addonecargo(self, material, description):
		"""
		Add one unit of the given material with the given description to cargo.
		There must be enough capacity to do this.
		"""
		assert material in self.MATERIALS
		remaining = self.getfreecapacity().get(material)
		assert remaining >= 1
		self.cargo[material].add(description)

	def removecargo(self, material, amount):
		"""
		Remove the given number of units of the given material to cargo.
		Which ones to remove (which descriptions) is chosen randomly.
		You cannot remove more than the existing amount.
		"""
		assert material in self.MATERIALS
		used = self.getusedcapacity().get(material)
		assert amount <= used
		names = random.sample(list(self.cargo[material]), amount)
		for name in names:
			self.cargo[material].remove(name)

	def addpersonnel(self, name):
		"""
		Add a person with the given name to personnel.
		"""
		assert name not in self.personnel
		self.personnel.add(name)

	def removepersonnel(self, name):
		"""
		Remove the person with the given name from personnel.
		"""
		assert name in self.personnel
		self.personnel.remove(name)

	def addstop(self, stoptime, stopname):
		"""
		Add a stop to the schedule.
		"""
		assert stoptime not in self.schedule.keys()
		assert stopname not in self.schedule.values()
		self.schedule[stoptime] = stopname

	def removeonestop(self):
		"""
		Remove some stop from the schedule.
		"""	
		assert len(self.schedule) > 0
		stoptime = random.choice(self.schedule.keys())
		self.schedule.pop(stoptime)

	@staticmethod
	def fromsecret(setofmaterials):
		"""
		Given a set of materials, return a new seed instance whose secret is
		that set of materials. The instance will have one empty car of each
		car type, and one full car for each requested type of material,
		plus one person, and one stop, which are the minimum.
		"""
		inst = RailyardInput()
		# add one empty car of each car type
		for car_type in RailyardInput.CAR_TYPES:
			inst.addcar(car_type, randomcarid())
		# add one extra full car for each requested cargo type 
		for m in setofmaterials:
			assert m in RailyardInput.MATERIALS, "Unknown material: {}".format(m)
			admissible_car_types = list(RailyardInput.MATERIAL_TO_CAR_TYPES[m])
			# don't pick Caboose (special case)
			if 'Caboose' in admissible_car_types:
				admissible_car_types.remove('Caboose')
			car_type = random.choice(admissible_car_types)
			# add the full car, except if Coal Car, of which there cannot be two
			# in that case we'll just fill up the already existing empty Coal Car
			if car_type != 'Coal Car':
				inst.addcar(car_type, randomcarid())
			(_, car_capacity) = RailyardInput.CAR_TYPES[car_type]
			for _ in range(car_capacity):
				inst.addonecargo(m, randomcargodescription())
		inst.addpersonnel(randompersonname())
		inst.addstop(randomstoptime(), randomstopname())
		return inst

	@staticmethod
	def fromrandomsecret():
		"""
		Return a new seed instance whose secret is a random set of materials,
		chosen with uniform probability from the power set. The instance will
		have one full car for each type of material, one person, and one stop.
		"""
		return RailyardInput.fromsecret(random.choice(list(powerset(RailyardInput.MATERIALS))))

	def __hash__(self):
		return hash((tuple(sorted([(k, frozenset(v)) for k, v in self.cars.iteritems()])),
			tuple(sorted([(k, frozenset(v)) for k, v in self.cargo.iteritems()])),
			frozenset(self.personnel),
			tuple(sorted([(k, v) for k, v in self.schedule.iteritems()]))))

class RailyardInputMutator(AppInputMutator):
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

class AddCar(RailyardInputMutator):
	"""
	Add one car of some type with a random ID.

	Fails if too many cars (arbitrary limit: 5).
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("AddCar.mutate()")
		if instance.getnumberofcars() == 5:
			return None
		candidates = instance.CAR_TYPES.keys()
		# Only one Coal Car is allowed in a train
		if 'Coal Car' in instance.cars:
			candidates.remove('Coal Car')
		# Only one Caboose is allowed in a train
		if 'Caboose' in instance.cars:
			candidates.remove('Caboose')
		car_type = random.choice(candidates)
		result = instance.clone()
		while True: # if it happens to be taken (very unlikely), just retry
			car_id = randomcarid()
			if not instance.caridinuse(car_id):
				break
		result.addcar(car_type, car_id)
		return result

class RemoveCar(RailyardInputMutator):
	"""
	Remove one car of some type. Note that this may possibly also remove some cargo
	if the current train capacity (after removing the car) cannot hold it.
	Thus this may alter the secret!

	Fails if there are no cars.
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("RemoveCar.mutate()")
		if instance.getnumberofcars() == 0:
			return None
		candidates = [cartype for (cartype, names) in instance.cars.iteritems() if len(names) > 0]
		car_type_to_remove = random.choice(candidates)
		material, car_capacity = instance.CAR_TYPES[car_type_to_remove]
		# How much of this material can we carry before removing the car?
		current_total_capacity = instance.gettotalcapacity()[material]
		# And how much are we actually using before removing the car?
		current_used_capacity = instance.getusedcapacity()[material]
		#print("Material: {}".format(material))
		#print("Car capacity: {}".format(car_capacity))
		#print("Current total capacity: {}".format(current_total_capacity))
		#print("Currently used capacity: {}".format(current_used_capacity))
		result = instance.clone()
		# Get rid of excess cargo, if any
		if current_used_capacity > current_total_capacity - car_capacity:
			excess = current_used_capacity - (current_total_capacity - car_capacity)
			#print("Excess: {}".format(excess))
			result.removecargo(material, excess)
		result.removeonecar(car_type_to_remove)
		return result
		

class AddCargo(RailyardInputMutator):
	"""
	Add some amount of some material.
	Amount is between 1 and the free capacity of that material.
	Uses a random description for each unit of cargo added.

	Fails if nothing can be added (all materials are at their max capacity).
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("AddCargo.mutate()")
		freecapacity = instance.getfreecapacity()
		candidates = [m for m in instance.MATERIALS if freecapacity[m] > 0]
		if not candidates:
			return None
		material = random.choice(candidates)
		capacity = freecapacity[material]
		amount = random.randint(1, capacity)
		result = instance.clone()
		for i in range(amount):
			description = randomcargodescription()
			result.addonecargo(material, description)
		return result


class RemoveCargo(RailyardInputMutator):
	"""
	Remove some amount of some material.

	Fails if nothing can be removed (there is no cargo at all).
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("RemoveCargo.mutate()")
		usedcapacity = instance.getusedcapacity()
		candidates = [m for m in instance.MATERIALS if usedcapacity[m] > 0]
		if not candidates:
			return None
		material = random.choice(candidates)
		used = usedcapacity[material]
		amount = random.randint(1, used)
		result = instance.clone()
		result.removecargo(material, amount)
		return result

class AddPerson(RailyardInputMutator):
	"""
	Add a person to the personnel.
	
	The person name is a random number of letters.
	(The chars should not matter, but the length might.)
	(The chars are randomized anyway, to avoid duplicate names.)

	Fails if too many people (arbitrary limit: 4).
	
	PEND: This is exploring both the number of people and the lengths of names.
	Should we be exploring those with separate mutators?
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("AddPerson.mutate()")
		if len(instance.personnel) == 4:
			return None
		while True: # if it happens to be taken (very unlikely), just retry
			name = randompersonname()
			if name not in instance.personnel:
				break
		result = instance.clone()
		result.addpersonnel(name)
		return result

class ReplacePerson(RailyardInputMutator):
	"""
	Replace a random person with another.
	
	The person name is a random number of letters.
	(The chars should not matter, but the length might.)
	(The chars are randomized anyway, to avoid duplicate names.)

	Fails if too many people (arbitrary limit: 4).
	
	PEND: This is exploring both the number of people and the lengths of names.
	Should we be exploring those with separate mutators?
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("AddPerson.mutate()")
		name2remove = random.choice(list(instance.personnel))
		result = instance.clone()
		result.removepersonnel(name2remove)
		while True: # if it happens to be taken (very unlikely), just retry
			name2add = randompersonname()
			if name2add not in instance.personnel:
				break
		result.addpersonnel(name2add)
		return result

class RemovePerson(RailyardInputMutator):
	"""
	Remove one person.

	Fails if there aren't at least two people (need at least 1 to run the train).
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("RemovePerson.mutate()")
		if len(instance.personnel) < 2:
			return None
		name = random.choice(list(instance.personnel))
		result = instance.clone()
		result.removepersonnel(name)
		return result

class AddStop(RailyardInputMutator):
	"""
	Add a stop to the schedule.
	
	The stop name is a random number of letters.
	(The chars should not matter, but the length might.)
	(The chars are randomized anyway, to avoid duplicate names.)

	Fails if too many stops (arbitrary limit: 4).
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("AddStop.mutate()")
		if len(instance.schedule) == 4:
			return None
		# Pick a time for the stop.
		# If it happens to be busy (very unlikely), just retry.
		while True:
			stoptime = randomstoptime()
			if stoptime not in instance.schedule.keys():
				break
		# Pick a name for the stop.
		# If it happens to be taken (very unlikely), just retry.
		while True:
			stopname = randomstopname()
			if stopname not in instance.schedule.values():
				break
		result = instance.clone()
		result.addstop(stoptime, stopname)
		return result

class RemoveStop(RailyardInputMutator):
	"""
	Removes a stop from the schedule.
	
	Fails if there is only one stop (need at least 2 to remove one).
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("RemoveStop.mutate()")
		if len(instance.schedule) < 2:
			return None
		result = instance.clone()
		result.removeonestop()
		return result
	
class ReplaceNamesSame(RailyardInputMutator):
	"""
	Replaces all names with new ones with same length.
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("ReplaceNamesSame.mutate()")
		personnel = set()
		for old_name in list(instance.personnel):
			name_len = len(old_name)
			while True: # if it happens to be taken (very unlikely), just retry
				name = randomword(name_len, name_len)
				if name not in personnel:
					personnel.add(name)
					break
		result = instance.clone()
		assert len(result.personnel) == len(personnel), "Length of previous and next personnel lists are not the same."
		result.personnel = personnel
		return result

class ReplaceNamesDifferent(RailyardInputMutator):
	"""
	Replaces all names with new ones with different lengths.
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("ReplaceNamesDifferent.mutate()")
		personnel = set()
		for i in range(len(list(instance.personnel))):
			while True: # if it happens to be taken (very unlikely), just retry
				name = randompersonname()
				if name not in personnel:
					personnel.add(name)
					break
		result = instance.clone()
		assert len(result.personnel) == len(personnel), "Length of previous and next personnel lists are not the same."
		result.personnel = personnel
		return result

class ReplaceStopLocations(RailyardInputMutator):
	"""
	Replaces stop locations, keeping times the same.
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("ReplaceStopLocations.mutate()")
		result = instance.clone()
		for key in result.schedule.iterkeys():
			result.schedule[key] = randomstopname()
		return result

class ReplaceStopLocationsDifferent(RailyardInputMutator):
	"""
	Replaces stop locations, keeping times and the stop name lengths the same.
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("ReplaceStopLocations.mutate()")
		result = instance.clone()
		for key in result.schedule.iterkeys():
			stopnamelen = len(result.schedule[key])
			result.schedule[key] = randomword(stopnamelen, stopnamelen)
		return result

class ShuffleStopLocations(RailyardInputMutator):
	"""
	Shuffle stop locations.
	"""
	@staticmethod
	def mutate(instance):
		if DEBUG_MUTATORS:
			print("ShuffleStopLocations.mutate()")
		result = instance.clone()
		values = list(result.schedule.values())
		random.shuffle(values)
		for i, key in enumerate(result.schedule.iterkeys()):
			result.schedule[key] = values[i]
		
		return result


