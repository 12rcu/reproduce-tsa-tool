import random, string, itertools

from itertools import chain, combinations

#Input, Mutators and Apps

##Airplan
from Airplan.airplannumcities import AirplanInput
from Airplan.airplannumcities import AddCity, AddConnection, RemoveCity, RemoveConnection
from Airplan.airplannumcities import IncreaseWeightsTimes10, DecreaseWeightsTimes10 
from Airplan.airplannumcities import IncreaseDensity, DecreaseDensity
from Airplan.airplannumcities import IncreaseWeightsBy1, DecreaseWeightsBy1, IncreaseWeightsBy10, DecreaseWeightsBy10
from Airplan.airplannumcities import IncrementWeight, IncrementWeightCol1, IncrementWeightCol2, IncrementWeightCol3, IncrementWeightCol4, IncrementWeightCol5, IncrementWeightCol6
from Airplan.airplannumcities import DecrementWeight, DecrementWeightCol1, DecrementWeightCol2, DecrementWeightCol3, DecrementWeightCol4, DecrementWeightCol5, DecrementWeightCol6
from Airplan.airplannumcities import IncrementWeightBy10, IncrementWeightCol1By10, IncrementWeightCol2By10, IncrementWeightCol3By10, IncrementWeightCol4By10, IncrementWeightCol5By10, IncrementWeightCol6By10
from Airplan.airplannumcities import DecrementWeightBy10, DecrementWeightCol1By10, DecrementWeightCol2By10, DecrementWeightCol3By10, DecrementWeightCol4By10, DecrementWeightCol5By10, DecrementWeightCol6By10
from Airplan.airplannumcities import IncrementWeightBy5, IncrementWeightCol1By5, IncrementWeightCol2By5, IncrementWeightCol3By5, IncrementWeightCol4By5, IncrementWeightCol5By5, IncrementWeightCol6By5
from Airplan.airplannumcities import DecrementWeightBy5, DecrementWeightCol1By5, DecrementWeightCol2By5, DecrementWeightCol3By5, DecrementWeightCol4By5, DecrementWeightCol5By5, DecrementWeightCol6By5

from Airplan.airplan_app import AirplanApp
#
##Tourplanner
from Tourplanner.tourplanner import TourPlannerInput, ReplaceOne, Shuffle, ReplaceTwo, ReplaceThree, ReplaceFour, ReplaceFive
from Tourplanner.tourplanner_app import TourPlannerApp
#
##Railyard
from Railyard.railyard import RailyardInput, AddCar, RemoveCar, AddCargo, RemoveCargo, AddPerson, RemovePerson, AddStop, RemoveStop
from Railyard.railyard import ReplaceNamesSame, ReplaceNamesDifferent, ReplaceStopLocations, ShuffleStopLocations
from Railyard.railyard_app import RailyardApp
#
##Gabfeed
from Gabfeed.gabfeed import GabfeedInput, AddOnes, RemoveOnes, ShuffleKey, AddFives, RemoveFives
from Gabfeed.gabfeed_app import GabfeedApp
#
##Powerbroker
from Powerbroker.powerbroker import PowerbrokerInput, IncreaseBudget, DecreaseBudget
from Powerbroker.powerbroker_app import PowerbrokerApp
#
##Bidpal
from Bidpal.bidpal import BidpalInput, IncreaseBid, DecreaseBid
from Bidpal.bidpal_app import BidpalApp
#
##Snapbuddy
from Snapbuddy.snapbuddy import SnapbuddyInput, ChangeCity
from Snapbuddy.snapbuddy_app import SnapbuddyApp

from Medpedia.medpedia import MedpediaInput, ChangeArticle
from Medpedia.medpedia_app import MedpediaApp

#Library for automatic runs
from tsa import AutoProfit

def powerset(iterable):
	"""
	powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
	"""
	xs = list(iterable)
	# note we return an iterator rather than a list
	return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

def run_airplan(airplan_version_name, num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit, new_children_size=100, runtime_per_step=None, mode=1):
	if airplan_version_name[0:len(airplan_version_name)-1] != 'airplan_':
		print("Name needs to be airplan_X where X is a number.") 
		return None
	inputs = []
	#Generating seed inputs where we generate 5 graphs per the number of 
	if mode == 1:
		for _ in range(5):
			for i in range(2,15):
				conns = []
				cities = [random_word(3,3) for _ in range(i)]
				conns_num = random.randint(0,int(i**2))
				if conns_num > 0:
					comb = list(itertools.product(cities,cities))
					random.shuffle(comb)

					conns = [None]*conns_num
					for j in range(conns_num):
						(src, dst) = comb[j]
						nums = [random_num(1,7) for _ in range(6)]
						conns[j] = (src, dst, nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])
				inputs.append(AirplanInput(set(cities),set(conns)))

	random.seed()

	mutators = [AddCity, AddConnection, RemoveCity, RemoveConnection, IncreaseWeightsTimes10, DecreaseWeightsTimes10, 
	IncreaseDensity, DecreaseDensity, IncreaseWeightsBy1, DecreaseWeightsBy1, IncreaseWeightsBy10, DecreaseWeightsBy10, 
	IncrementWeight, IncrementWeightCol1, IncrementWeightCol2, IncrementWeightCol3, IncrementWeightCol4, IncrementWeightCol5, IncrementWeightCol6, 
	DecrementWeight, DecrementWeightCol1, DecrementWeightCol2, DecrementWeightCol3, DecrementWeightCol4, DecrementWeightCol5, DecrementWeightCol6, 
	IncrementWeightBy10, IncrementWeightCol1By10, IncrementWeightCol2By10, IncrementWeightCol3By10, IncrementWeightCol4By10, IncrementWeightCol5By10, IncrementWeightCol6By10, 
	DecrementWeightBy10, DecrementWeightCol1By10, DecrementWeightCol2By10, DecrementWeightCol3By10, DecrementWeightCol4By10, DecrementWeightCol5By10, DecrementWeightCol6By10, 
	IncrementWeightBy5, IncrementWeightCol1By5, IncrementWeightCol2By5, IncrementWeightCol3By5, IncrementWeightCol4By5, IncrementWeightCol5By5, IncrementWeightCol6By5, 
	DecrementWeightBy5, DecrementWeightCol1By5, DecrementWeightCol2By5, DecrementWeightCol3By5, DecrementWeightCol4By5, DecrementWeightCol5By5, DecrementWeightCol6By5]

	grouped_mutators = mutators
	ports = [8443]
	exp_name = "{}_test".format(airplan_version_name)
	app = AirplanApp(airplan_version_name, exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=True, calcTime=False, grouped_mutators=grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=False, mutation_weighing_on=mut_weigh_on, 
		stop_criterion_on = stop_crit, new_children_size=new_children_size, runtime_per_step=runtime_per_step)

	ap.netrunner()

def run_railyard(num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit, new_children_size=None, runtime_per_step=300.0):
	inputs = []
	for i,secret in enumerate(list(powerset(RailyardInput.MATERIALS))):
		inputs.append(RailyardInput.fromsecret(secret))

	mutators = [AddCar, RemoveCar, AddCargo, RemoveCargo, AddPerson, RemovePerson, AddStop, RemoveStop, 
		ReplaceNamesSame, ReplaceNamesDifferent, ReplaceStopLocations] #, ShuffleStopLocations]

	grouped_mutators = [[AddCar, RemoveCar], [AddCargo, RemoveCargo], [AddPerson, RemovePerson], [AddStop, RemoveStop], 
		[ReplaceNamesSame], [ReplaceNamesDifferent], [ReplaceStopLocations]] #, [ShuffleStopLocations]]

	ports = [4567]
	exp_name = 'railyard_test'
	app = RailyardApp('railyard', 'railyard_test', ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name=exp_name, calcSpace=True, calcTime=False, grouped_mutators=grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on,
		stop_criterion_on = stop_crit, new_children_size=new_children_size, runtime_per_step=runtime_per_step)

	ap.netrunner()

def run_tp(num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit):
	places = ['Boston', 'Worcester', 'Springfield','Lowell','Cambridge',
			'New Bedford','Brockton','Quincy','Lynn','Fall River',
			'Newton','Lawrence','Somerville','Framingham','Haverhill',
			'Waltham','Malden','Brookline','Plymouth','Medford',
			'Taunton','Chicopee','Weymouth','Revere','Peabody']
	inputs = [None]*5
	inputs[0] = TourPlannerInput([places[0], places[3], places[21], places[6], places[4]])
	inputs[1] = TourPlannerInput([places[11], places[2], places[5], places[22], places[14]])
	inputs[2] = TourPlannerInput([places[4], places[19], places[7], places[20], places[6]])
	inputs[3] = TourPlannerInput([places[13], places[23], places[1], places[7], places[18]])
	inputs[4] = TourPlannerInput([places[1], places[4], places[7], places[17], places[8]])

	mutators = [ReplaceOne, ReplaceTwo, ReplaceThree, ReplaceFour, ReplaceFive, Shuffle]
	grouped_mutators = [[ReplaceOne], [Shuffle], [ReplaceTwo], [ReplaceThree], [ReplaceFour], [ReplaceFive]]
	
	ports = [8989]
	exp_name = 'tourplanner_test'

	app = TourPlannerApp('tour_planner', exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=False, calcTime=True, grouped_mutators = grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()

def run_gabfeed(program_name, num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit):
	inputs = [None]*17
	for i in range(len(inputs)): # 0-64
		inputs[i] = GabfeedInput(i*4)

	mutators = [AddOnes, RemoveOnes, AddFives, RemoveFives, ShuffleKey]
	ports = [8080]

	grouped_mutators = [[AddOnes, RemoveOnes], [AddFives, RemoveFives], [ShuffleKey]]

	exp_name = '{}_test'.format(program_name)
	app = GabfeedApp(program_name, exp_name, ports)
	
	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=False, calcTime=True, grouped_mutators = grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()

def run_powerbroker(program_name, num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit):
	inputs = [None]*4
	for i in range(len(inputs)): # 10-500
		inputs[i] = PowerbrokerInput((i+1)*100)

	mutators = [IncreaseBudget, DecreaseBudget]
	ports = [9000, 9001]
	exp_name = '{}_test'.format(program_name)

	app = PowerbrokerApp(program_name, exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=False, calcTime=True,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()

def run_bidpal(program_name, num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit):
	inputs = [None]*4
	for i in range(len(inputs)): # 10-500
		inputs[i] = BidpalInput(100*(i+1))

	mutators = [IncreaseBid, DecreaseBid]
	ports = [8000, 8001]
	exp_name = '{}_test'.format(program_name)

	app = BidpalApp(program_name, exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=False, calcTime=True,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()

def run_snapbuddy(num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit):
	city2bssids = {}
	citylist = []

	with open('./Snapbuddy/cities.txt', 'r') as citiesfile:
		for line in citiesfile:
			line = line.strip()
			if '","' in line:
				name, bssids = line.split('","')
				name = name.strip('"')
				bssids = bssids.strip('"')
				city2bssids[name] = bssids
				citylist.append(name)


	program_name = 'snapbuddy_1'
	inputs = [SnapbuddyInput(citylist[i]) for i in range(10)]

	mutators = [ChangeCity]
	ports = [8080]
	exp_name = '{}_test'.format(program_name)

	app = SnapbuddyApp(program_name, exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=True, calcTime=False, 
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()

def run_medpedia(num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit, new_children_size=1000, runtime_per_step=None):
	random.seed(42)
	
	inputs = []
	for i in range(500):
		inputs.append(MedpediaInput())
	
	random.seed()

	mutators = [ChangeArticle]
	grouped_mutators = [[ChangeArticle]]

	ports = [8443]
	exp_name = 'ase_init_medpedia_test'
	app = MedpediaApp('medpedia', exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name=exp_name, calcSpace=True, calcTime=False, grouped_mutators=grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on,
		stop_criterion_on = stop_crit, new_children_size=new_children_size, runtime_per_step=runtime_per_step)

	print('STARTING!')
	ap.netrunner()

def random_word(min,max):
	n = random.randint(min,max)
	return ''.join(random.choice(string.ascii_uppercase) for _ in range(n))

def random_num(min,max):
	n = random.randint(min,max)
	return int(''.join(random.choice(string.digits) for _ in range(n)))

if __name__ == "__main__":
	rpi_init_time = 100
	rpi_init_space = 1
	num_steps = 30
	rpi_est_time = False
	rpi_est_space = False

	mutation_weighing = True
	stop_criterion = True

	print(""" 
	Experiment Setup Parameters
	Number of Steps: {}
	For Time Experiments  - RPI: {}
	For Space Experiments - RPI: {}
	Mutation Weighing: {}
	Stop Criterion: {}""".format(num_steps, rpi_init_time, rpi_init_space, mutation_weighing, stop_criterion))
	
	#print("Profit vs. AP run!")
	for x_name in ['airplan_3', 'airplan_5', 'airplan_2']:
		run_airplan(x_name, num_steps, rpi_init_space, rpi_est_space, mutation_weighing, stop_criterion, new_children_size=100, runtime_per_step=None, mode=1)
	for x_name in ['gabfeed_1', 'gabfeed_2', 'gabfeed_5']:
		run_gabfeed(x_name, num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)
	for x_name in ['powerbroker_1', 'powerbroker_2', 'powerbroker_4']:
		run_powerbroker(x_name, num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)
	for x_name in ['bidpal_2', 'bidpal_1']:
		run_bidpal(x_name, num_steps, rpi_init_time,  rpi_est_time,  False, stop_criterion)
	
	run_snapbuddy(num_steps, 10,  rpi_est_space,  False, stop_criterion) #Has one mutator, does not need mutation weighing
	run_tp(num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)

	#run_medpedia(num_steps=100, rpi_init=36, rpi_est=rpi_est_space, mut_weigh_on=False, stop_crit=stop_criterion, new_children_size=500, runtime_per_step=None)




