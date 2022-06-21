import matplotlib
matplotlib.use('Agg') #Doesn't work on NUCs without this, visualizations crash
#import sniffer as tsa
import matplotlib.pyplot as plt
import numpy as np
import random, os, string, itertools, pprint, time, math, copy

from itertools import chain, combinations
from operator import itemgetter

from numpy.random import choice

#Input, Mutators and Apps

##Airplan
from airplannumcities import AirplanInput
from airplannumcities import AddCity, AddConnection, RemoveCity, RemoveConnection, IncreaseWeights
from airplannumcities import IncreaseWeights, DecreaseWeights, IncreaseDensity, DecreaseDensity, IncrementWeight, DecrementWeight
from airplan_app import AirplanApp
#
##Tourplanner
#from tourplanner import TourPlannerInput, ReplaceOne, Shuffle
#from tp_app import TourPlannerApp
#
##Railyard
#from railyard import RailyardInput, AddCar, RemoveCar, AddCargo, RemoveCargo, AddPerson, RemovePerson, AddStop, RemoveStop
#from railyard_app import RailyardApp
#
##Gabfeed
#from gabfeed import GabfeedInput, AddOnes, RemoveOnes, ShuffleKey, AddFives, RemoveFives
#from gabfeed_app import GabfeedApp
#
##Powerbroker
#from powerbroker import PowerbrokerInput, IncreaseBudget, DecreaseBudget
#from powerbroker_app import PowerbrokerApp
#
##Bidpal
#from bidpal import BidpalInput, IncreaseBid, DecreaseBid
#from bidpal_app import BidpalApp
#
##Snapbuddy
#from snapbuddy import SnapbuddyInput, ChangeCity
#from snapbuddy_app import SnapbuddyApp

#from litmedia_media import LitmediaMediaApp, LitmediaMediaInput
#from litmedia_bookmark import LitmediaBookmarkApp, LitmediaBookmarkInput, AddBookmark, RemoveBookmark
#from litmedia_pref import LitmediaPrefApp, LitmediaPrefInput

#Profit toolkit for sniffing, quantifying, etc.
from tsa import Transform, Sniffer, Visualize, Utils, AutoProfit

ids = [
		"1834086304",
		"2465446604",
		"9494681214",
		"5209264327",
		"9792180639",
		"7229195145",
		"1995613618",
		"4415062522",
		"5263798949",
		"0598177976",
		"5261570329",
		"8718161110",
		"3349216164",
		"3525872986",
		"3686118510",
		"7856688579",
		"8194314790",
		"2946153851",
		"0076454431",
		"5220481394",
		"9712860842",
		"2220727573",
		"2366367190",
		"9125849804",
		"3952240174",
		"5603928083",
		"8356588116",
		"5453757607",
		"5725137409",
		"8991051767",
		"8803150110",
		"4717358681",
		"2463254924",
		"8890317599",
		"5401485149",
		"0208846669",
		"4920194067",
		"9200653002",
		"3947313242",
		"5629159750",
		"1794239028",
		"7343094649",
		"1807114918",
		"0197194347",
		"7234037189",
		"2417231431",
		"8588363329",
		"9204098298",
		"3611745479",
		"8527304688",
		"0278573372",
		"1665806262",
		"4571878820",
		"1135891360",
		"6793132242",
		"5397010150",
		"9308792687",
		"3643058070",
		"5548358717",
		"7465922295",
		"0927977595",
		"9609022969",
		"3944929373",
		"5527111568",
		"3673137701"]

def run_litmedia_prefs(app_name='litmedia_1'):
	prefs = [
		('philosophy',['sad']),
		('science',['soccer']),
		('sports',['basketball']),
		('politics',['bill']),
		('self_help',['retro']),
		('gaming',['soccer','sad']),
		('sports',['football']),
		('self_help',['abstract']),
		('adventure',['shakespearean']),
		('sports',['soccer','meta','shakespearean']),
		('food',['writing']),
		('food',['sad','sweet']),
		('academia',['math','beautiful']),
		('nature',['animals','outdoors']),
		('adventure',['abstract']),
		('art',['puzzle']),
		('food',['retro']),
		('nature',['animals','outdoors']),
		('food',['conservative']),
		('adventure',['love']),
		('adventure',['outdoors']),
		('romance',['justice']),
		('romance',['sweet']),
		('philosophy',['horror']),
		('nature',['outdoors']),
		('romance',['love']),
		('politics',['justice']),
		('self_help',['travel']),
		('academia',['beautiful']),
		('adventure',['journalism','writing']),
		('gaming',['retro']),
		('romance',['writing']),
		('romance',['writing']),
		('gaming',['Sad']),
		('nature',['horror']),
		('science',['love']),
		('food',['outdoors']),
		('philosophy',['conservative']),
		('science',['exciting','love','retro']),
		('art',['sad']),
		('politics',['writing']),
		('self_help',['beautiful']),
		('nature',['football']),
		('adventure',['bill']),
		('academia',['puzzle']),
		('sports',['democratic','bill']),
		('adventure',['shakespearean']),
		('nature',['retro','writing','conservative']),
		('politics',['writing']),
		('self_help',['sweet']),
		('academia',['math']),
		('adventure',['abstract']),
		('gaming',['shakespearean']),
		('science',['sad','exciting']),
		('art',['justice']),
		('food',['meta']),
		('self_help',['democratic']),
		('politics',['conservative']),
		('art',['travel','exciting']),
		('academia',['journalism']),
		('philosophy',['beautiful']),
		('romance',['justice','puzzle','meta']),
		('adventure',['math']),
		('politics',['sweet']),
		('food',['journalism','sweet','love']),
		]

	#prefs = list(set(prefs))

	inputs = []

	#All bookmarks with one book
	for (i, ident) in enumerate(prefs):
		inputs.append(LitmediaPrefInput([(ident[0],ident[1],55)]))

	for i in range(250):
		prefs_selected = random.sample(prefs, 5)

		pref_list = [(pref[0], pref[1], i+1) for (i,pref) in enumerate(prefs_selected)]

		inputs.append(LitmediaPrefInput(pref_list))

	##10 random versions of bookmarks from 2 to N-1
	#for i in range(10):
	#	for j in range(2, len(ids)):
	#		bookmark_size_j = random.sample(ids, j)
	#		inputs.append(LitmediaBookmarkInput(bookmark_size_j))

	#Bookmarks with all the books and none of the books.
	#inputs.append(LitmediaBookmarkInput([]))
	#inputs.append(LitmediaBookmarkInput(ids[:]))

	num_steps = 1
	rpi_init = 5
	rpi_est = False
	mut_weigh_on = False
	grouped_mutators = None
	stop_crit = False

	ports = [8443]
	mutators = []

	n = 25

	for i in xrange(0,len(inputs),n):
		inputs_subset = inputs[i:i+n]

		if app_name == 'litmedia_1':
			exp_name = 'litmedia_1_getmedia_Q32_TIME_ITER{}_{}x{}RPIx2USERS'.format(i, len(inputs), rpi_init)
		else:
			exp_name = 'litmedia_2_getmedia_Q01_TIME_ITER{}_{}x{}RPIx2USERS'.format(i, len(inputs), rpi_init)

		app = LitmediaPrefApp(app_name, exp_name, ports)

		ap = AutoProfit(inputs_subset, mutators, app, ports, exp_name=exp_name, calcSpace=False, calcTime=True, grouped_mutators=grouped_mutators,
			number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, 
			stop_criterion_on = stop_crit)

		ap.netrunner()



def run_litmedia_getbookmarks(app_name='litmedia_2'):

	#inputs = [None] * len(ids)
	
	inputs = []
	
	#All bookmarks with one book
	#for (i, ident) in enumerate(ids):
	#	inputs[i] = LitmediaBookmarkInput([ident])

	for i in range(300):
		bookmark_size_j = random.sample(ids, 5)
		inputs.append(LitmediaBookmarkInput(bookmark_size_j))

	#10 random versions of bookmarks from 2 to N-1
	#for i in range(10):
	#	for j in range(2, len(ids)):
	#		bookmark_size_j = random.sample(ids, j)
	#		inputs.append(LitmediaBookmarkInput(bookmark_size_j))

	#Bookmarks with all the books and none of the books.
	#inputs.append(LitmediaBookmarkInput([]))
	#inputs.append(LitmediaBookmarkInput(ids[:]))

	num_steps = 1
	rpi_init = 5
	rpi_est = False
	mut_weigh_on = False
	grouped_mutators = None
	stop_crit = False

	ports = [8443]
	mutators = []

	n=25

	for i in xrange(0,len(inputs), n):
		new_inps = inputs[i:i+n]

		if app_name == 'litmedia_1':
			exp_name = 'litmedia_1_getmedia_Q56_SPACETIME_ITER{}_{}x{}RPIx2USERS'.format(i, len(new_inps), rpi_init)
		else:
			exp_name = 'litmedia_2_getmedia_Q58_SPACETIME_ITER{}_{}x{}RPIx2USERS'.format(i, len(new_inps), rpi_init)

		app = LitmediaBookmarkApp(app_name, exp_name, ports)

		ap = AutoProfit(new_inps, mutators, app, ports, exp_name=exp_name, calcSpace=True, calcTime=True, grouped_mutators=grouped_mutators,
			number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, 
			stop_criterion_on = stop_crit)

		ap.netrunner(inf_quant=False)

def run_litmedia_getmedia():
	ids = [
		"1834086304",
		"2465446604",
		"9494681214",
		"5209264327",
		"9792180639",
		"7229195145",
		"1995613618",
		"4415062522",
		"5263798949",
		"0598177976",
		"5261570329",
		"8718161110",
		"3349216164",
		"3525872986",
		"3686118510",
		"7856688579",
		"8194314790",
		"2946153851",
		"0076454431",
		"5220481394",
		"9712860842",
		"2220727573",
		"2366367190",
		"9125849804",
		"3952240174",
		"5603928083",
		"8356588116",
		"5453757607",
		"5725137409",
		"8991051767",
		"8803150110",
		"4717358681",
		"2463254924",
		"8890317599",
		"5401485149",
		"0208846669",
		"4920194067",
		"9200653002",
		"3947313242",
		"5629159750",
		"1794239028",
		"7343094649",
		"1807114918",
		"0197194347",
		"7234037189",
		"2417231431",
		"8588363329",
		"9204098298",
		"3611745479",
		"8527304688",
		"0278573372",
		"1665806262",
		"4571878820",
		"1135891360",
		"6793132242",
		"5397010150",
		"9308792687",
		"3643058070",
		"5548358717",
		"7465922295",
		"0927977595",
		"9609022969",
		"3944929373",
		"5527111568",
		"3673137701"]

	inputs = [None] * len(ids)
	for (i, ident) in enumerate(ids):
		inputs[i] = LitmediaMediaInput(ident)

	num_steps = 1
	rpi_init = 20
	rpi_est = False
	mut_weigh_on = False
	grouped_mutators = None
	stop_crit = False

	ports = [8443]
	mutators = []

	exp_name = 'litmedia_2_getmedia_Q50_TIME_{}x{}'.format(len(inputs), rpi_init)
	app = LitmediaMediaApp('litmedia_2', exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name=exp_name, calcSpace=False, calcTime=True, grouped_mutators=grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, 
		stop_criterion_on = stop_crit)

	ap.netrunner()

#if __name__ == '__main__':
#	for app_name in ['litmedia_1','litmedia_2']:
#		run_litmedia_prefs(app_name)
		#run_litmedia_getbookmarks(app_name)
	#run_litmedia_getmedia()

#OLD VERSIONS:
def run_airplan(airplan_version_name, num_steps=30, rpi_init=1, rpi_est=False, mut_weigh_on=False, stop_crit=False):
	if airplan_version_name[0:len(airplan_version_name)-1] != 'airplan_':
		print "Name needs to be airplan_X where X is a number." 
		return None
	inputs = []
	for i in range(2,15):
		cities = [random_word(3,3) for _ in range(i)]
		conns = []

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

	mutators = [AddCity, AddConnection, RemoveCity, RemoveConnection, IncreaseWeights, DecreaseWeights, IncreaseDensity, DecreaseDensity, IncrementWeight, DecrementWeight]
	grouped_mutators = [[AddCity, RemoveCity], [AddConnection, RemoveConnection], [IncreaseWeights, DecreaseWeights],
		[IncreaseDensity, DecreaseDensity], [IncrementWeight, DecrementWeight]]
	ports = [8443]
	exp_name = "{}_5m_test".format(airplan_version_name)
	app = AirplanApp(airplan_version_name, exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=True, calcTime=False, grouped_mutators=grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()

def run_railyard(num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit):
	inputs = []
	for i,secret in enumerate(list(powerset(RailyardInput.MATERIALS))):
		#print(secret)
		inputs.append(RailyardInput.fromsecret(secret))

	mutators = [AddCar, RemoveCar, AddCargo, RemoveCargo, AddPerson, RemovePerson, AddStop, RemoveStop]
	grouped_mutators = [[AddCar, RemoveCar], [AddCargo, RemoveCargo], [AddPerson, RemovePerson], [AddStop, RemoveStop]]

	ports = [4567]
	exp_name = 'railyard_5m_test'
	app = RailyardApp('railyard', 'railyard_test', ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name=exp_name, calcSpace=True, calcTime=False, grouped_mutators=grouped_mutators,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

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

	mutators = [ReplaceOne, Shuffle]
	grouped_mutators = [[ReplaceOne],[Shuffle]]
	
	ports = [8989]
	exp_name = 'tour_planner_5m_test'

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

	exp_name = '{}_5m_test'.format(program_name)
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
	exp_name = '{}_5m_test'.format(program_name)

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
	exp_name = '{}_5m_test'.format(program_name)

	app = BidpalApp(program_name, exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=False, calcTime=True,
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()

def run_snapbuddy(num_steps, rpi_init, rpi_est, mut_weigh_on, stop_crit):
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


	program_name = 'snapbuddy_1'
	inputs = [SnapbuddyInput(citylist[i]) for i in range(10)]

	mutators = [ChangeCity]
	ports = [8080]
	exp_name = '{}_5m_test'.format(program_name)

	app = SnapbuddyApp(program_name, exp_name, ports)

	ap = AutoProfit(inputs, mutators, app, ports, exp_name, calcSpace=True, calcTime=False, 
		number_of_run_steps=num_steps, repetitions_per_input=rpi_init, rpi_estimation_on=rpi_est, mutation_weighing_on=mut_weigh_on, stop_criterion_on = stop_crit)

	ap.netrunner()




def random_word(min,max):
	n = random.randint(min,max)
	return ''.join(random.choice(string.ascii_uppercase) for _ in range(n))

def random_num(min,max):
	n = random.randint(min,max)
	return int(''.join(random.choice(string.digits) for _ in range(n)))

def main():
	rpi_init_time = 1
	rpi_init_space = 1
	num_steps = 13
	rpi_est_time = False
	mutation_weighing = False
	rpi_est_space = False
	stop_criterion = False
	
	#for x_name in ['gabfeed_1', 'gabfeed_2','gabfeed_5']:
	#	run_gabfeed(x_name, num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)	
	for x_name in ['airplan_3', 'airplan_5']:
		run_airplan(x_name)
	#for x_name in ['powerbroker_1','powerbroker_2','powerbroker_4']:
	#	run_powerbroker(x_name, num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)
	#for x_name in ['bidpal_2', 'bidpal_1']:
	#	run_bidpal(x_name, num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)
	#run_railyard(num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)
	#run_snapbuddy(num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)
	#run_tp(	num_steps, rpi_init_time,  rpi_est_time,  mutation_weighing, stop_criterion)



if __name__=='__main__':
	main()
