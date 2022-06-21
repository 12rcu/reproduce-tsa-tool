import random, copy, uuid, string, itertools
from collections import Counter

from mutation import AppInput, AppInputMutator

city2bssids = {}
citylist = []

with open('cities', 'r') as citiesfile:
	for line in citiesfile:
		line = line.strip()
		if '","' in line:
			name, bssids = line.split('","')
			name = name.strip('"')
			bssids = bssids.strip('"')
			city2bssids[name] = bssids
			citylist.append(name)

class SnapbuddyInput(AppInput):

	def __init__(self, cityname=''):
		super(SnapbuddyInput, self).__init__()
		# Cities are a list in the app but we're modeling it as a set of strings.
		if cityname == '':
			cityname = random.choice(citylist)
		self.cityname = cityname

	def clone(self):
		"""
		Return a clone of self.
		"""
		return copy.deepcopy(self)

	def getsecret(self):
		"""
		Return the value of the secret for this input.
		"""
		return self.cityname
	
	def getbssid(self):
		return city2bssids[self.cityname]

	def __str__(self):
		"""
		Return plain text representation.
		"""
		return self.cityname

	def __hash__(self):
		return hash(self.cityname)

class SnapbuddyInputMutator(AppInputMutator):
	pass

class ChangeCity(SnapbuddyInputMutator):
	@staticmethod
	def mutate(instance):
		newcity = random.choice(citylist)

		result = instance.clone()
		result.cityname = newcity
		return result
