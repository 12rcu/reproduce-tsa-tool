import random, copy, string, itertools

from tsa.mutation import AppInput, AppInputMutator

MAX_INT = 2147483647

class AirplanInput(AppInput):

	def __init__(self, cities = set(['A','B','C']), connections=set()):
		super(AirplanInput, self).__init__()
		# Cities are a list in the app but we're modeling it as a set of strings.
		self.cities = cities
		assert len(list(cities))>1
		# We model the connections as an 8 tuple (2 city names, 6 integers as weights).
		self.connections = connections

	def clone(self):
		"""
		Return a clone of self.
		"""
		return copy.deepcopy(self)

	def getsecret(self):
		"""
		Return the value of the secret for this input.
		"""
		return '{:02d}'.format(len(self.cities))

	def __str__(self):
		"""
		Return plain text representation.
		"""
		strs = (
			'cities:      {}'.format(self.cities),
			'connections: {}'.format(self.connections),
		)
		return '\n'.join(strs)

	def getnumberofcities(self):
		"""
		Return the total number of cities.
		"""
		return len(self.cities)

	def getnumberofconnections(self):
		"""
		Return the total number of connections.
		"""
		return len(self.connections)

	def addcity(self, city_name):
		"""
		Add a city with a given name.
		"""
		assert len(city_name) <= 3 and len(city_name) >= 1 and city_name not in self.cities
		self.cities.add(city_name)
	
	def removecity(self, city_name):
		"""
		Remove a city with a given name.
		"""
		assert len(city_name) <= 3 and len(city_name) >= 1 and city_name in self.cities
		self.cities.remove(city_name)

	def addconnection(self, connection):
		"""
		Add a connection which is a tuple with 8 length.
		"""
		assert len(connection) == 8
		assert connection[0] in self.cities
		assert connection[1] in self.cities
		for (src,dst,_,_,_,_,_,_) in self.connections:
			if src == connection[0] and dst == connection[1]:
				return
		self.connections.add(connection)

	def removeconnection(self, src_city, dst_city):
		"""
		Remove a connection from the edges.
		"""
		assert src_city in self.cities
		assert dst_city in self.cities
		conn_copy = [] + list(self.connections)

		for connection in conn_copy:
			(src,dst,a,b,c,d,e,f) = connection
			if src == src_city and dst == dst_city:
				self.connections.remove(connection)
				break

	def removeconnections(self, city):
		"""
		Remove all connections that contain the city name that is passed.
		"""

		assert city in self.cities
		
		conn_copy = [] + list(self.connections)
		for connection in conn_copy:
			(src,dst,a,b,c,d,e,f) = connection
			if src == city or dst == city:
				self.connections.remove(connection)

	def __hash__(self):
		x=(tuple(self.cities), tuple(self.connections))
		return hash(x)

class AirplanInputMutator(AppInputMutator):
	pass

class AddCity(AirplanInputMutator):
	"""
	Add one city to the set of cities.

	Fails if there are too many cities (arbitrary limit: 14).
	"""
	@staticmethod
	def mutate(instance):
		#print("AddCity.mutate()")
		if instance.getnumberofcities() >= 14:
			return None
		
		city_name = ""
		while (city_name == "" or city_name in instance.cities):
			name_length = random.randint(1,3)
			city_name = "".join(random.choice(string.uppercase) for _ in range(name_length))
		
		result = instance.clone()
		result.addcity(city_name)
		return result

class AddConnection(AirplanInputMutator):
	"""
	Adds a connection to the connection set.

	Fails if nothing can be added (all edges between all nodes are there).
	"""
	@staticmethod
	def mutate(instance):
		#print("AddConnection.mutate()")
		available = instance.getnumberofcities()**2 - instance.getnumberofconnections()
		if available <= 0:
			return None
		
		#Selecting src/dst pairs randomly
		c = list(instance.cities)
		all_combs = set(itertools.product(c,c))
		current_combs = set([(tup[0],tup[1]) for tup in list(instance.connections)])
		possible_combs = list(all_combs - current_combs)
		if len(possible_combs) <= 0:
			return None
		(src_city, dst_city) = random.choice(possible_combs)
		
		#Setting up weights
		nums = [0] * 6
		for i in range(6):
			num_length = random.randint(1,7)
			nums[i] = int("".join(random.choice(string.digits) for _ in range(num_length)))
		
		conn = (src_city, dst_city, nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])
		
		result = instance.clone()
		result.addconnection(conn)
		return result

class RemoveCity(AirplanInputMutator):
	"""
	Removes a random city from the set of cities.
	It also removes the connections related to that city.

	Fails if there are too less cities (arbitrary limit: 2).
	"""
	@staticmethod
	def mutate(instance):
		#print("RemoveCity.mutate()")
		if instance.getnumberofcities() <= 2:
			return None

		city_name = random.choice(list(instance.cities))

		result = instance.clone()
		result.removeconnections(city_name)
		result.removecity(city_name)
		return result

class RemoveConnection(AirplanInputMutator):
	"""
	Removes a connection from the connection set.

	Returns None if nothing can be removed (the edge set is empty).
	"""
	@staticmethod
	def mutate(instance):
		#print("RemoveConnection.mutate()")
		available = instance.getnumberofconnections()
		if available <= 0:
			return None
		connections = instance.connections
		(src_city, dst_city, _, _, _, _, _, _) = random.choice(list(connections))
		
		result = instance.clone()
		result.removeconnection(src_city, dst_city)
		return result

class IncreaseDensity(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("IncreaseDensity.mutate()")
		available = instance.getnumberofconnections()
		total = instance.getnumberofcities()**2
		density = 0.2
		num_nodes_added = min(int(total/5), total-available)
		if num_nodes_added <= 0:
			return None

		result = instance.clone()
		for i in range(num_nodes_added):
			temp = AddConnection.mutate(result)
			if temp is None:
				return result
			else:
				result = temp

		return result

class DecreaseDensity(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("DecreaseDensity.mutate()")
		available = instance.getnumberofconnections()
		total = instance.getnumberofcities()**2
		density = 0.2
		num_nodes_removed = min(int(total/5), available)
		if num_nodes_removed <= 0:
			return None

		result = instance.clone()
		for i in range(num_nodes_removed):
			temp = RemoveConnection.mutate(result)
			if temp is None:
				return result
			else:
				result = temp
		return result

class DecreaseWeightsTimes10(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("DecreaseWeights.mutate()")
		m = 10
		conn_list = list(instance.connections)
		for (i,(src,dst,a,b,c,d,e,f)) in enumerate(conn_list):
			a = max(0, int(a/m))
			b = max(0, int(b/m))
			c = max(0, int(c/m))
			d = max(0, int(d/m))
			e = max(0, int(e/m))
			f = max(0, int(f/m))
			conn_list[i] = (src,dst,a,b,c,d,e,f)

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncreaseWeightsTimes10(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("IncreaseWeights.mutate()")
		m = 10
		conn_list = list(instance.connections)
		for (i,(src,dst,a,b,c,d,e,f)) in enumerate(conn_list):
			a = min(MAX_INT, int(a*m))
			b = min(MAX_INT, int(b*m))
			c = min(MAX_INT, int(c*m))
			d = min(MAX_INT, int(d*m))
			e = min(MAX_INT, int(e*m))
			f = min(MAX_INT, int(f*m))
			conn_list[i] = (src,dst,a,b,c,d,e,f)

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecreaseWeightsBy1(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("DecreaseWeights.mutate()")
		m = 1
		conn_list = list(instance.connections)
		for (i,(src,dst,a,b,c,d,e,f)) in enumerate(conn_list):
			a = max(0, int(a-m))
			b = max(0, int(b-m))
			c = max(0, int(c-m))
			d = max(0, int(d-m))
			e = max(0, int(e-m))
			f = max(0, int(f-m))
			conn_list[i] = (src,dst,a,b,c,d,e,f)

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncreaseWeightsBy1(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("IncreaseWeights.mutate()")
		m = 1
		conn_list = list(instance.connections)
		for (i,(src,dst,a,b,c,d,e,f)) in enumerate(conn_list):
			a = min(MAX_INT, int(a+m))
			b = min(MAX_INT, int(b+m))
			c = min(MAX_INT, int(c+m))
			d = min(MAX_INT, int(d+m))
			e = min(MAX_INT, int(e+m))
			f = min(MAX_INT, int(f+m))
			conn_list[i] = (src,dst,a,b,c,d,e,f)

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecreaseWeightsBy10(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("DecreaseWeights.mutate()")
		m = 10
		conn_list = list(instance.connections)
		for (i,(src,dst,a,b,c,d,e,f)) in enumerate(conn_list):
			a = max(0, int(a-m))
			b = max(0, int(b-m))
			c = max(0, int(c-m))
			d = max(0, int(d-m))
			e = max(0, int(e-m))
			f = max(0, int(f-m))
			conn_list[i] = (src,dst,a,b,c,d,e,f)

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncreaseWeightsBy10(AirplanInputMutator):
	"""
	"""
	@staticmethod
	def mutate(instance):
		#print("IncreaseWeights.mutate()")
		m = 10
		conn_list = list(instance.connections)
		for (i,(src,dst,a,b,c,d,e,f)) in enumerate(conn_list):
			a = min(MAX_INT, int(a+m))
			b = min(MAX_INT, int(b+m))
			c = min(MAX_INT, int(c+m))
			d = min(MAX_INT, int(d+m))
			e = min(MAX_INT, int(e+m))
			f = min(MAX_INT, int(f+m))
			conn_list[i] = (src,dst,a,b,c,d,e,f)

		result = instance.clone()
		result.connections = set(conn_list)
		return result

#Providing *very specific* mutators as an example

class IncrementWeight(AirplanInputMutator):
	"""
	Increment one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = random.randint(0,5)
		modifier = [0]*6
		if tup[fine_index+2] < MAX_INT:
			modifier[fine_index] = 1
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol1(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 0
		modifier = [0]*6
		if tup[fine_index+2] < MAX_INT:
			modifier[fine_index] = 1
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol2(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 1
		modifier = [0]*6
		if tup[fine_index+2] < MAX_INT:
			modifier[fine_index] = 1
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol3(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 2
		modifier = [0]*6
		if tup[fine_index+2] < MAX_INT:
			modifier[fine_index] = 1
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol4(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 3
		modifier = [0]*6
		if tup[fine_index+2] < MAX_INT:
			modifier[fine_index] = 1
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol5(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 4
		modifier = [0]*6
		if tup[fine_index+2] < MAX_INT:
			modifier[fine_index] = 1
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol6(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 5
		modifier = [0]*6
		if tup[fine_index+2] < MAX_INT:
			modifier[fine_index] = 1
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeight(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = random.randint(0,5)
		modifier = [0]*6
		if tup[fine_index+2] > 0:
			modifier[fine_index] = -1

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol1(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 0
		modifier = [0]*6
		if tup[fine_index+2] > 0:
			modifier[fine_index] = -1

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol2(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 1
		modifier = [0]*6
		if tup[fine_index+2] > 0:
			modifier[fine_index] = -1

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol3(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 2
		modifier = [0]*6
		if tup[fine_index+2] > 0:
			modifier[fine_index] = -1

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol4(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 3
		modifier = [0]*6
		if tup[fine_index+2] > 0:
			modifier[fine_index] = -1

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 4
		modifier = [0]*6
		if tup[fine_index+2] > 0:
			modifier[fine_index] = -1

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol6(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 5
		modifier = [0]*6
		if tup[fine_index+2] > 0:
			modifier[fine_index] = -1

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightBy10(AirplanInputMutator):
	"""
	Increment one of the weights by 10.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = random.randint(0,5)
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 10)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol1By10(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 0
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 10)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol2By10(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 1
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 10)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol3By10(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 2
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 10)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol4By10(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 3
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 10)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol5By10(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 4
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 10)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol6By10(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 5
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 10)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightBy10(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = random.randint(0,5)
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 10)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol1By10(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 0
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 10)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol2By10(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 1
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 10)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol3By10(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 2
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 10)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol4By10(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 3
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 10)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol5By10(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 4
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 10)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol6By10(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 10
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 5
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 10)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightBy5(AirplanInputMutator):
	"""
	Increment one of the weights by 5.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = random.randint(0,5)
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 5)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol1By5(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 0
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 5)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol2By5(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 1
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 5)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol3By5(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 2
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 5)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol4By5(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 3
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 5)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol5By5(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 4
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 5)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class IncrementWeightCol6By5(AirplanInputMutator):
	"""
	Increment one of the weights of first column by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 5
		modifier = [0]*6
		modifier[fine_index] = min(MAX_INT-tup[fine_index+2], 5)
		
		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])
		
		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightBy5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = random.randint(0,5)
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 5)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol1By5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 0
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 5)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol2By5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 1
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 5)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol3By5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 2
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 5)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol4By5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 3
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 5)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol5By5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 4
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 5)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result

class DecrementWeightCol6By5(AirplanInputMutator):
	"""
	Decrement one of the weights by 1.
	"""
	@staticmethod
	def mutate(instance):
		m = 5
		conn_list = list(instance.connections)

		if len(conn_list) == 0:
			return None

		random_index = random.randint(0,len(conn_list)-1)
		tup = conn_list[random_index]

		fine_index = 5
		modifier = [0]*6
		modifier[fine_index] = -1 * min(tup[fine_index+2], 5)

		conn_list[random_index] = (tup[0], tup[1], tup[2]+modifier[0], tup[3]+modifier[1], tup[4]+modifier[2], tup[5]+modifier[3], tup[6]+modifier[4], tup[7]+modifier[5])

		result = instance.clone()
		result.connections = set(conn_list)
		return result
