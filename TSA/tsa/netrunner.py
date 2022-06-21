from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg') #Doesn't work on NUCs without this, visualizations cause crash with ssh connection
import matplotlib.pyplot as plt
import numpy as np
import random, time, math, copy

from itertools import chain, combinations
from operator import itemgetter
from numpy.random import choice
import math

#Profit toolkit for sniffing, quantifying, etc.
from tsa import Transform, Utils

def powerset(iterable):
	"""
	powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
	"""
	xs = list(iterable)
	# note we return an iterator rather than a list
	return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

def save_inputs(folder_loc, inputs):
	for i,inp in enumerate(inputs):
		f = open('{}/{:03d}.txt'.format(folder_loc,i),'w')
		f.write(str(inp))
		f.write('\n')
		f.close()

class AutoProfit(object):
	def __init__(self, inputs, mutators, app, ports, exp_name = 'autotsa_test', calcSpace=True, calcTime=True, dry=False, 
		initial_div_size = None, new_children_size = None, number_of_run_steps = 30, repetitions_per_input = 20, rpi_estimation_on = False, 
		mutation_weighing_on=False, grouped_mutators=None, stop_criterion_on=False, runtime_per_step=300.0, mutator_weights=None):
		#Variable about initial and generated inputs
		self.all_inputs = inputs[:] #This var keeps all inputs generated or passed as seeds
		self.new_inputs = inputs[:] #This var keeps all inputs that would be run

		#Variable that keeps the measurements over the network
		self.interactions = []

		#These are needed for the input generation and multi-step runs.
		self.mutators = mutators
		self.app = app
		self.ports = ports
		self.exp_name = exp_name
		#Variable about exploring Space or Timing Side Channels
		self.calcSpace = calcSpace
		self.calcTime = calcTime
		#This var is about dry running, if True, we only generate new inputs and save them, don't run it.
		self.dry = dry
		#Variables about initial diversity and new input sizes. If None (as default), these will be estimated automatically for a 5 minute run.
		self.initial_div_size = initial_div_size
		self.new_children_size = new_children_size

		self.number_of_run_steps = number_of_run_steps
		self.repetitions_per_input = repetitions_per_input

		self.runtime_per_step = runtime_per_step #seconds

		if self.new_children_size is None and self.runtime_per_step is None:
			self.new_children_size = 100
		
		#RPI Estimation Option and parameters
		self.rpi_estimation_on    = rpi_estimation_on

		self.rpi_error_bound      = 0.0125
		self.rpi_confidence_const = 1.96 ##1.96 #95% confidence
		self.rpi_inputs_to_run    = 5
		self.rpi_eval_value       = 40
		self.rpi_num_features     = 10
		self.rpi_bound            = 20
		
		#Mutation Weighing Option and parameters
		self.mutation_weighing_on = mutation_weighing_on
		self.grouped_mutators = grouped_mutators
		self.mutator_weights = None
		self.mw_num_features = 5

		self.extra_mutator_weights = mutator_weights

		#Stop Criterion Option and parameters
		self.stop_criterion_on           = stop_criterion_on
		#self.stop_criterion_bound       = 0.02 # 0.02 bits
		self.stop_criterion_bound        = 0.01 # 1%, using percentage based stop criterion
		self.stop_criterion_length       = 5
		self.stop_criterion_num_features = 3
	
	def gen_new_inputs(self, inputs, mutators, num_new_inps=100, min_mutation=1, max_mutation=10, weight_vector=None):
		"""
		This function takes the inputs, mutates them to new inputs and returns only the new inputs.

		Inputs:
			inputs: The list of AppInput objects which are used for mutation.
			num_new_inputs: The number of new inputs generated, default value is 100.
			min_mutation: The minumum number of mutators applied to the input to generate a new input, default value is 1.
			max_mutation: The maximum number of mutators applied to the input to generate a new input, default value is 10.
			weight_vector: The list of weights which represent the importance of each mutator. 
				Its default value is None which makes the weights equal for each mutator. 
		Outputs:
			new_inputs: The list of new inputs which are mutated from the inputs list.
		"""
		
		hashmap = dict()
		for inp in inputs:
			hashmap[hash(inp)] = inp

		#Save each input at the end
		orig_inputs = inputs[:]
		new_inputs = []
		if weight_vector is None:
			weight_vector = [1.0/len(mutators)]*len(mutators)

		orig_weight_vector = weight_vector[:]
		orig_seed_list = []

		while len(new_inputs) < num_new_inps:
			
			if len(orig_inputs) == 0 and len(new_inputs) > 0:
				return new_inputs, orig_seed_list
			if len(orig_inputs) == 0 and len(new_inputs) == 0:
				print("We've tried all mutators and all inputs and we can't generate new inputs.")
				print("Either we have explored all input space or you need to write better mutators to explore remaining input space.")
				return None, orig_seed_list
			
			#Select a random input in list, select a random number of mutations (1-10),

			orig_seed = random.choice(orig_inputs)
			seed = copy.deepcopy(orig_seed)
			if min_mutation != max_mutation:
				mutation_limit = random.randint(min_mutation,max_mutation)
			else:
				mutation_limit = min_mutation
			candidates = None
			if type(mutators) is list:
				candidates = mutators[:]
			else:
				candidates = list(mutators)

			#NEW INPUT GENERATION LOOP
			weight_vector = orig_weight_vector[:]
			successful_mutation_num = 0
			
			while successful_mutation_num < mutation_limit:
				if len(candidates) == 0:
					break
				
				mutatorclass_index = choice(range(len(candidates)), 1, p=weight_vector)[0]
				mutatorclass = candidates[mutatorclass_index]
				saved_seed = copy.deepcopy(seed)
				temp = mutatorclass.mutate(seed)
				
				if temp is None or (successful_mutation_num == mutation_limit-1 and hash(temp) in hashmap):
					# mutator failed, don't try it anymore on this input
					#print mutatorclass
					
					candidates.remove(mutatorclass)
					del weight_vector[mutatorclass_index]
					weight_vector = [x/sum(weight_vector) for x in weight_vector]
					if len(candidates) == 0:
						# no more mutators left
						break
				else:
					# mutator succeeded!
					successful_mutation_num += 1
					seed = temp
					orig_seed_list.append( (saved_seed, mutatorclass, copy.deepcopy(seed)) )
			
			if successful_mutation_num > 0 and hash(orig_seed) != hash(seed):
				hashmap[hash(seed)] = seed
				new_inputs.append(seed)
				orig_inputs.append(orig_seed)
			elif orig_seed in orig_inputs:
				orig_inputs.remove(orig_seed)
			else:
				print("Original seed can't be mutated but it's not in inputs list?")
				print("Successful_mutation_num:{}".format(successful_mutation_num))

		#PRINTING INITIAL DIVERSITY
		#for i,inp in enumerate(inputs):
		#	print '{}:\n{}'.format(i, inp)

		print("Length of new inputs: {}".format(len(new_inputs)))
		print("Length of seed list: {}".format(len(orig_seed_list)))
		return new_inputs, orig_seed_list

	def gen_new_inputs_mw(self, inputs, mutators, num_new_inps=100, min_mutation=1, max_mutation=10, weight_vector=None):
		"""
		This function takes the inputs, mutates them to new inputs and returns only the new inputs.
		
		Inputs:
			inputs: The list of AppInput objects which are used for mutation.
			num_new_inputs: The number of new inputs generated, default value is 100.
			min_mutation: The minumum number of mutators applied to the input to generate a new input, default value is 1.
			max_mutation: The maximum number of mutators applied to the input to generate a new input, default value is 10.
			weight_vector: The list of weights which represent the importance of each mutator. 
				Its default value is None which makes the weights equal for each mutator. 
		Outputs:
			new_inputs: The list of new inputs which are mutated from the inputs list.
		"""
		#Save each input at the end
		orig_inputs = inputs[:]
		new_inputs = []
		if weight_vector is None:
			weight_vector = [1.0/len(mutators)]*len(mutators)

		orig_weight_vector = weight_vector[:]

		num_different_secrets = 0

		for i in range(num_new_inps):
			#Select a random input in list, select a random number of mutations (1-10),
			seed = random.choice(orig_inputs)
			seed_orig = copy.deepcopy(seed)
			mutation_limit = random.randint(min_mutation,max_mutation)
			candidates = None
			if type(mutators) is list:
				candidates = mutators[:]
			else:
				candidates = list(mutators)

			#print "Candidates:{}".format(candidates)
			weight_vector = orig_weight_vector[:]
			for i in range(mutation_limit):
				if len(candidates) == 0:
					break
				
				mutatorclass_index = choice(range(len(candidates)), 1, p=weight_vector)[0]
				#mutatorclass_index = mutatorclass_index[0]
				mutatorclass = candidates[mutatorclass_index]
				#print mutatorclass
				temp = mutatorclass.mutate(seed)
				if temp is None:
					# mutator failed, don't try it anymore on this input
					#print mutatorclass
					candidates.remove(mutatorclass)
					del weight_vector[mutatorclass_index]
					weight_vector = [x/sum(weight_vector) for x in weight_vector]
					if len(candidates) == 0:
						# no more mutators left
						break
				else:
					# mutator succeeded!
					seed = temp

			if seed_orig.getsecret() != seed.getsecret():
				num_different_secrets += 1
			new_inputs.append(seed)

		#PRINTING INITIAL DIVERSITY
		#for i,inp in enumerate(inputs):
		#	print '{}:\n{}'.format(i, inp)
		return (new_inputs, num_different_secrets)

	def extract_stats(self, inputs):
		sec_dist = dict()
		for i in inputs:
			sec = i.getsecret()
			if sec in sec_dist:
				sec_dist[sec] = sec_dist[sec]+1
			else:
				sec_dist[sec] = 1
		return sec_dist

	def netrunner(self, inf_quant=True, use_phases=True, option='kde-dynamic'):
		"""
		This function contains the main execution loop which is repeating the process of
		input generation, running the inputs over the system while sniffing the network traffic 
		and analyzing the captured traces.

		Inputs:
		inf_quant: Determining whether the run will include trace analysis and leakage quantification. Default value is True.
			Disabling this only generates traces via input generation which may be analyzed later.
		use_phases: Boolean variable determining whether trace alignment is used for trace analysis. Default value is True.
			It slows down the trace analysis, so the users may choose to have results without alignment.
		option: This argument sets the information leakage quantification method. Default value is 'kde-dynamic'. The possible options are:
				normal             - Quantifying information by modeling the data distribution with Normal distributions.
				hist               - Quantifying information by modeling the data distribution with Histograms.
				kde                - Quantifying information by modeling the data distribution with Kernel Density Estimation (KDE).
				kde-dynamic        - Quantifying information by modeling the data distribution with KDE with bandwidth selection using cross-validation.
				gmm                - Quantifying information by modeling the data distribution with Bayesian Gaussian Mixture Model.
				leakiest-integer   - Quantifying information using Leakiest tool, integer mode. Used for discrete features (like packet sizes).
				leakiest-real      - Quantifying information using Leakiest tool, real mode. Used for continuous features (like packet timings).
				fbleau             - Quantifying information using F-BLEAU tool.
		"""
		bdwidth = 0.25
		quant_mode_list = ['normal', 'hist', 'kde', 'kde-fixed', 'kde-dynamic', 'gmm', 'leakiest-integer', 'leakiest-real', 'fbleau']
		if option not in quant_mode_list:
			print("Option needs to be one of {}".format(quant_mode_list))
			return None
		
		if option == 'kde':
			bdwidth = None
		
		#Execution params
		intrs = []

		exptime_list = []
		quanttime_list = []
		samplenum_list = []
		if not self.dry:
			self.app.launchcontainers()
			self.app.startserver()

		space_leakage_history = []
		time_leakage_history = []
		max_leakage_history = []

		for run_count in range(0, self.number_of_run_steps+1):
			print("Run #{0:02d}/#{1}".format(run_count, self.number_of_run_steps))
			plot = run_count%5
			
			"""
			#Secret statistics over inputs
			all_sec_dict = extract_stats(all_inputs)
			new_sec_dict = extract_stats(new_inputs)
			str_all = pprint.pformat(all_sec_dict)
			str_new = pprint.pformat(new_sec_dict)
			print("New inputs secret distribution:\n{}".format(str_new))
			print("All inputs secret distribution:\n{}".format(str_all))
			
			if i %10 == 1 or i == run_limit-1:
				tup_list = all_sec_dict.items()
				tup_list.sort(key=lambda x: x[0])
				sec_list, count_list = zip(*tup_list)
				sec_list = list(sec_list)
				count_list = list(count_list)
				total_count = sum(count_list)
				
				fig, ax = plt.subplots()
				y_pos = np.arange(len(sec_list))
				percentages = [float(x)/total_count for x in count_list]
				ax.bar(y_pos, percentages, align='center', alpha=0.5)
				ax.set_ylim([0.0,1.0])
				plt.xticks(y_pos, sec_list)
				plt.ylabel('Percentage over all secrets')
				plt.title('Secret distribution over inputs')
				fig.savefig('{}_bar_chart_step_{:03d}.png'.format(exp_name, i), dpi=600)
				plt.close(fig)
			"""

			if not self.dry: #Not dry means we're running it on the system, dry run means we're just generating new items.
				exptime_start = time.time()

				self.app.startsniffer()

				print("LENGTH NEW INPUTS", len(self.new_inputs))
				print("RPI:", self.repetitions_per_input)

				#RUNNING NEW INPUTS
				self.app.run_inputs(self.new_inputs, self.repetitions_per_input) #Silence the printing of packets
				
				self.app.sniffer.clean_marked_interactions()
				temp_intrs = self.app.finishexperiment('step{:02d}'.format(run_count))
				self.app.sniffer.interactions = None
				self.app.sniffer.pcap = None
				#self.app.sniffer.processed_intrs = None
				self.interactions = self.interactions + [[Utils.convert2packet(p) for p in temp_intr] for temp_intr in temp_intrs]

				#folder_name = app.folder_name + '{:02d}_stage_inputs'.format(i)
				#if not os.path.exists(folder_name):
				#	 os.makedirs(folder_name)
				#save_inputs(folder_name, new_inputs)

				exptime_end = time.time()
				
				if inf_quant:
					pcap_filename = '{0}_run_{1:02d}.pcap'.format(self.exp_name, run_count)

					#Calling the sniffer with the interactions, returns the leakage which we will not care for now.
					#QUANTIFICATION PART
					(all_space_features, all_time_features, all_space_tags, all_time_tags, space_all_leakage, time_all_leakage) = Utils.process_all(ports=self.ports, 
						pcap_filename=pcap_filename, interactions=self.interactions, 
						intervals=None, calcSpace=self.calcSpace, calcTime=self.calcTime, use_phases=use_phases, quant_mode=option, window_size=bdwidth, plot=plot, dry_mode=False)
					#use_phases=True
					
					#PRINTING STATISTICS
					print("Number of interactions: {}".format(len(self.interactions)))

					quanttime_end = time.time()

					exptime   = float(exptime_end)   - float(exptime_start)
					quanttime = float(quanttime_end) - float(exptime_end)

					exptime_ps   = float(exptime)   / (self.repetitions_per_input*len(self.new_inputs))
					quanttime_ps = float(quanttime) / (self.repetitions_per_input*len(self.all_inputs))

					runstring = "{} Run {}/{}: ".format(self.exp_name, run_count, self.number_of_run_steps)
					print(runstring + " Samples per input for this step   :       {}".format(self.repetitions_per_input))
					print(runstring + " Number of inputs run for this step:       {}".format(len(self.new_inputs)))
					print(runstring + " Number of inputs run in total     :       {}".format(len(self.all_inputs)))
					print(runstring + " Number of traces generated for this step: {}".format(len(temp_intrs)))
					print(runstring + " Number of traces generated in total     : {}".format(len(self.interactions)))
					print(runstring + " Total running time:  {:0.3f} s".format(exptime))
					print(runstring + " Total quantify time: {:0.3f} s".format(quanttime))
					print(runstring + " Running        time per sample {:0.3f} s".format(exptime_ps))
					print(runstring + " Quantification time per sample {:0.3f} s".format(quanttime_ps))
				
					exptime_list.append(exptime)
					quanttime_list.append(quanttime)

					print(runstring + "Exptime list", [xs for xs in exptime_list])
					print(runstring + "Quanttime list", [xs for xs in quanttime_list])

					print('List version, leakage results for run {}/{}'.format(run_count, self.number_of_run_steps))

					secret_leakage = float(np.log2(len(set(Transform.rd_secrets(self.interactions))))) 
					#Calculate total possible information leakage from number of secrets, take np.log2 of it.
					max_leakage_history.append(secret_leakage)
		
					if self.calcSpace:
						print("Leakage for step {}: {}".format(run_count,space_all_leakage))
						space_leakage_history.append(space_all_leakage)
						print('Leakage history in bits: {}'.format(space_leakage_history))
						lp = [[(leakage) for (leakage, tag) in l[:10]] for l in space_leakage_history]
						print("Leakage in list form:"+ str(lp))

						#Plot 1
						#fig, ax = plt.subplots()
						#ax.plot(range(1,len(max_leakage_history)+1), max_leakage_history, '--') # CURRENT LEAKAGE PLOT, EXCLUDED FOR NOW

						#lp = map(list, zip(*lp))
						#for lx in lp:
						#	#print(lx)
						#	ax.plot(range(0,len(lp[0])), lx)
						#ax.set_xlabel("Run steps")
						#ax.set_ylabel("Leakage in bits")
						#ax.set_title("Experiment: {}, Secret information: {:0.3f} bits".format(self.exp_name, secret_leakage))

						#fig.savefig(self.app.folder_name + '{}_step{}_bits.png'.format(self.exp_name,run_count),dpi=1200)
						#plt.close(fig)

						#Plot 2
						#fig, ax = plt.subplots()

						#ax.plot(range(1,len(max_leakage_history)+1), max_leakage_history, '--') # CURRENT LEAKAGE PLOT, EXCLUDED FOR NOW

						#for lx in lp:
						#	ax.plot(range(0,len(lp[0])), [val/ml for (ml,val) in zip(max_leakage_history,lx)])
						#ax.set_xlabel("Run steps")
						#ax.set_ylabel("Leakage in percentages")
						#ax.set_title("Experiment: {}, Secret information: {:0.3f} bits".format(self.exp_name, secret_leakage))

						#fig.savefig(self.app.folder_name + '{}_step{}_percent.png'.format(self.exp_name,run_count),dpi=1200)
						#plt.close(fig)

						#STOP CRITERION
						if self.stop_criterion_on and len(space_leakage_history) >= self.stop_criterion_length:
							lp_lastfive = [[float(leakage)/max_leakage_history[-1] for (leakage, tag) in l[:self.stop_criterion_num_features]] for l in space_leakage_history[-self.stop_criterion_length:]]
							stop_criterion = True

							tp_lastfive = map(list, zip(*lp_lastfive))
							for lx in tp_lastfive:
								if (max(lx) - min(lx)) > self.stop_criterion_bound:
									stop_criterion = False
									break

							if stop_criterion:
								print("STOPPING because top {} features did not have any change within the bound of {} for last {} steps.".format(self.stop_criterion_num_features,self.stop_criterion_bound, self.stop_criterion_length))
								#return None

					if self.calcTime:
						print("Leakage for step {}: {}".format(run_count,time_all_leakage))
						time_leakage_history.append(time_all_leakage)
						print('Leakage history in bits: {}'.format(time_leakage_history))
						lp = [[leakage for (leakage, tag) in l[:10]] for l in time_leakage_history]
						print("Leakage in list form:"+ str(lp))
				
						#Plot 1
						#fig, ax = plt.subplots()
						#ax.plot(range(1,len(max_leakage_history)+1),max_leakage_history, '--')

						#lp = map(list, zip(*lp))
						#for lx in lp:
						#	ax.plot(range(0,len(lp[0])),lx)
						#ax.set_xlabel("Run steps")
						#ax.set_ylabel("Leakage in bits")
						#ax.set_title("Experiment: {}, total information: {:0.3f} bits".format(self.exp_name, secret_leakage))

						#fig.savefig(self.app.folder_name + '{}_step{}_bits.png'.format(self.exp_name,run_count),dpi=1200)
						#plt.close(fig)

						#Plot 2
						#fig, ax = plt.subplots()

						#for lx in lp:
						#	ax.plot(range(0,len(lp[0])),[val/ml for (ml,val) in zip(max_leakage_history,lx)])
						#ax.set_xlabel("Run steps")
						#ax.set_ylabel("Leakage in percentages")
						#ax.set_title("Experiment: {}, total information: {:0.3f} bits".format(self.exp_name, secret_leakage))

						#fig.savefig(self.app.folder_name + '{}_step{}_percent.png'.format(self.exp_name,run_count),dpi=1200)
						#plt.close(fig)

						#STOP CRITERION
						if self.stop_criterion_on and len(time_leakage_history) >= self.stop_criterion_length:
							lp_lastfive = [[float(leakage)/max_leakage_history[-1] for (leakage, tag) in l[:self.stop_criterion_num_features]] for l in time_leakage_history[-self.stop_criterion_length:]]
							stop_criterion = True

							tp_lastfive = map(list, zip(*lp_lastfive))
							for lx in tp_lastfive:
								if (max(lx) - min(lx)) > self.stop_criterion_bound:
									stop_criterion = False
									break

							if stop_criterion:
								print("STOPPING because top {} features did not have any change within the bound of {} for last {} steps.".format(self.stop_criterion_num_features,self.stop_criterion_bound, self.stop_criterion_length))
								#return None

				#If it's the step after running initial diversity, we're going to call rpi estimation with leakage results of top X features.
				if self.rpi_estimation_on and run_count==1:
					if self.calcSpace:
						leakage_list = space_leakage_history[-1]
					elif self.calcTime:
						leakage_list = time_leakage_history[-1]
					leakage_list.sort(key=itemgetter(0), reverse=True)
					leakage_sublist = leakage_list[:self.rpi_num_features]

					self.find_rpi(leakage_sublist)

					if self.runtime_per_step is not None:
						self.new_children_size = int(math.floor(self.runtime_per_step/(exptime_ps*self.repetitions_per_input)))
					if self.new_children_size == 0:
						self.new_children_size = 1
			
				#If we're weighing mutations and it's the step *after* initial diversity, send the variables and results to the mutation_weighing()
				#Get the results, update the weights and use the weights from now on!
				if self.mutation_weighing_on and run_count==1:
					initial_results = None
					initial_tag_results = None
					initial_feature_results = []
					
					if self.calcTime:
						initial_results = time_leakage_history[-1][:self.mw_num_features]
					if self.calcSpace:
						initial_results = space_leakage_history[-1][:self.mw_num_features]
					print("Initial Results: {}".format(initial_results))
					self.mutation_weighing_discovery(initial_results=initial_results)
					
					#print("{} New Weights: {}".format(self.exp_name, zip(mutators,weights)))
					#print("New mutators: {}".format(mutators))

			#GENERATING NEW INPUTS
			if run_count == self.number_of_run_steps:
				print("RUN COUNT EQUALS PREDETERMINED NUMBER OF STEPS. RUNCOUNT={}, PREDET.NUMSTEPS={}".format(run_count, self.number_of_run_steps))
				self.app.removecontainers()
				return None

			if run_count == 0: #If it's the first step, we will generate initial diversity which mutates more than usual.
				if self.runtime_per_step is not None:
					self.new_children_size = int(math.floor(self.runtime_per_step/(exptime_ps*self.repetitions_per_input)))
					print("{} seconds corresponds to {} inputs at each step.".format(self.runtime_per_step, self.new_children_size))
				if self.new_children_size == 0:
					self.new_children_size = 1
				self.new_inputs,_ = self.gen_new_inputs(self.all_inputs, self.mutators, self.new_children_size, 1, 1, self.mutator_weights)
				if self.new_inputs is None:
					print("No new inputs generated, terminating early.")
					return None
				self.all_inputs = self.all_inputs + self.new_inputs
			else:
				self.new_inputs,_ = self.gen_new_inputs(self.all_inputs, self.mutators, self.new_children_size, 1, 1, self.mutator_weights)
				if self.new_inputs is None:
					print("No new inputs generated, terminating early.")
					return None
				self.all_inputs = self.all_inputs + self.new_inputs

		if not self.dry:
			self.app.removecontainers()

		return None


	def mutation_weighing_discovery(self, initial_results=None):
		if self.extra_mutator_weights is not None:
			self.mutator_weights = self.extra_mutator_weights[:]
			return None

		results = [None]*len(self.grouped_mutators)
		new_inputs_overall = []
		all_intrs_overall = []
		
		new_weights = [0.0]*len(self.grouped_mutators)
		
		related_tags = [tag for (leakage, tag) in initial_results]
		initial_results = [leakage for (leakage, tag) in initial_results]
		#feature_val_orig = initial_results[2]
		#feature_val_results = [None]*len(self.grouped_mutators)
		#new_vals_diff = [0]*len(self.grouped_mutators)
		new_secs_diff = [0]*len(self.grouped_mutators)
		
		#initial_results = initial_results[0]
		
		print(str(related_tags))
		print(str(initial_results))
		
		for (i, group) in enumerate(self.grouped_mutators):
			print("Testing for mutator {}/{}: {}".format(i+1, len(self.grouped_mutators), group))

			new_inputs_size = int(np.ceil(float(self.new_children_size)/len(self.grouped_mutators)))

			print("Generating {} inputs to test mutators {}".format(new_inputs_size, group))

			#TODO Need to pair old and new inputs so that we can do the analysis
			#TODO Need to pair old inputs to interactions so that we don't need to rerun the old inputs *again*

			new_inputs, seed_result_pairs = self.gen_new_inputs(inputs=self.all_inputs, mutators=group, num_new_inps=new_inputs_size, min_mutation=1, max_mutation=1)
			new_inputs_overall = new_inputs_overall[:] + new_inputs[:]

			inputs_to_run = [seed for (seed,_,_) in seed_result_pairs] + [res for (_,_,res) in seed_result_pairs]
			
			print("NEW INPUTS LENGTH: {}".format(len(new_inputs)))
			print("NUM INPUTS TO RUN: {}".format(len(inputs_to_run)))
			
			
			new_inputs_size = len(inputs_to_run)/2
			
			self.app.startsniffer()
			#RUNNING NEW INPUTS
			self.app.run_inputs(inputs_to_run, self.repetitions_per_input) #Silence the printing of packets

			self.app.sniffer.clean_marked_interactions()
			temp_intrs = self.app.finishexperiment('dim_{}'.format(i+1))
			processed_intrs = [[Utils.convert2packet(p) for p in temp_intr] for temp_intr in temp_intrs]
			#all_intrs_overall = all_intrs_overall[:] + processed_intrs[:]
			#intrs = self.interactions + processed_intrs

			#TODO Fix this stupid file naming from Utils.process_all, it chops last 5 chars.
			pcap_filename = '{}_run_dim_{}.pcap'.format(self.exp_name, i+1)

			print("Results for mutators {}/{}: {}".format(i+1, len(self.grouped_mutators), group))

			#Calling the sniffer with the interactions, returns the leakage which we will not care for now.
			(dim_space_features, dim_time_features, dim_space_tags, dim_time_tags, _, _) = Utils.process_all(ports=self.ports,
				dry_mode=True, pcap_filename=pcap_filename, interactions=processed_intrs, intervals=None, 
				calcSpace=self.calcSpace, calcTime=self.calcTime, use_phases=True, quant_mode='normal')

			print("Number of NEW interactions: {}".format(len(processed_intrs)))
			#print "Number of interactions: {}".format(len(intrs))

			if self.calcSpace:
				results[i] = []
				for j,tag in enumerate(related_tags):
					for k, t in enumerate(dim_space_tags):
						if t == tag:
							weight_counter = 0.0
			
							if self.repetitions_per_input==1:
								for l in range(new_inputs_size):
									if dim_space_features[k][l] != dim_space_features[k][l+new_inputs_size]:
										weight_counter += 0.5
									if inputs_to_run[l].getsecret() != inputs_to_run[l+new_inputs_size].getsecret():
										weight_counter += 0.5
							else:
								mean1=0.0
								mean2=0.0
								max_val = max(dim_space_features[k]) - min(dim_space_features[k])
								if max_val == 0.0:
									max_val = 1.0

								for m in range(self.repetitions_per_input):
									for l in range(new_inputs_size):
										mean1 += dim_space_features[k][l+m*2*new_inputs_size]
										mean2 += dim_space_features[k][l+(m*2+1)*new_inputs_size]
										if inputs_to_run[l].getsecret()	!= inputs_to_run[l+new_inputs_size].getsecret():
											weight_counter += 0.5
								mean1 = mean1/(max_val*self.repetitions_per_input)
								mean2 = mean2/(max_val*self.repetitions_per_input)
								weight_counter = abs(mean1 - mean2)
							print("DIMENSION, TAG, WEIGHT: {}, {}, {}".format(group, tag, weight_counter))
							new_weights[i] = new_weights[i] + weight_counter


			if self.calcTime:
				results[i] = []
				for j, tag in enumerate(related_tags):
					for k, t in enumerate(dim_time_tags):
						if t == tag:
							weight_counter = 0.0
							if self.repetitions_per_input==1:
								for l in range(new_inputs_size):
									if dim_time_features[k][l] != dim_time_features[k][l+new_inputs_size]:
										weight_counter += 0.5
									if inputs_to_run[l].getsecret()	!= inputs_to_run[l+new_inputs_size].getsecret():
										weight_counter += 0.5
							else:
								mean1=0.0
								mean2=0.0
								max_val = max(dim_time_features[k]) - min(dim_time_features[k])
								if max_val == 0.0:
									max_val = 1.0

								for m in range(self.repetitions_per_input):
									for l in range(new_inputs_size):
										mean1 += dim_time_features[k][l+m*2*new_inputs_size]
										mean2 += dim_time_features[k][l+(m*2+1)*new_inputs_size]
									if inputs_to_run[l].getsecret()	!= inputs_to_run[l+new_inputs_size].getsecret():
										weight_counter += 0.5

								mean1 = mean1/(max_val*self.repetitions_per_input)
								mean2 = mean2/(max_val*self.repetitions_per_input)
								weight_counter = abs(mean1 - mean2)
							
							print("DIMENSION, TAG, WEIGHT: {}, {}, {}".format(group, tag, weight_counter))
							new_weights[i] = new_weights[i] + weight_counter

		print("{} New Weights with all features: {}".format(self.exp_name, zip(self.grouped_mutators, new_weights)))

		full_weights = []
		new_mutators = []
		for i, group in enumerate(self.grouped_mutators):
			for el in group:
				new_mutators.append(el)
				full_weights.append(new_weights[i])

		new_weights = full_weights[:]
		if sum(new_weights) == 0:
			new_weights = [1.0/len(new_weights)]*len(new_weights)
		else:
			new_weights = [x/sum(new_weights) for x in new_weights]

		self.mutator_weights = new_weights[:]
		self.mutators = new_mutators[:]
		#self.interactions = self.interactions + all_intrs_overall
		#self.all_inputs = self.all_inputs + new_inputs_overall
		print("{} New Weights: {}".format(self.exp_name, zip(new_mutators,new_weights)))

		return None


	def mutation_weighing(self, initial_results=None):
		if self.extra_mutator_weights is not None:
			self.mutator_weights = self.extra_mutator_weights[:]
			return None

		results = [None]*len(self.grouped_mutators)
		new_inputs_overall = []
		all_intrs_overall = []
		
		new_weights = [0.0]*len(self.grouped_mutators)
		
		related_tags = [tag for (leakage, tag) in initial_results]
		initial_results = [leakage for (leakage, tag) in initial_results]
		#feature_val_orig = initial_results[2]
		#feature_val_results = [None]*len(self.grouped_mutators)
		#new_vals_diff = [0]*len(self.grouped_mutators)
		new_secs_diff = [0]*len(self.grouped_mutators)
		
		#initial_results = initial_results[0]
		
		print(str(related_tags))
		print(str(initial_results))
		
		for (i, group) in enumerate(self.grouped_mutators):
			print("Testing for mutator {}/{}: {}".format(i+1, len(self.grouped_mutators), group))

			new_inputs_size = max(int(np.ceil(float(self.new_children_size)/len(self.grouped_mutators))),10)

			print("Generating {} inputs to test mutators {}".format(new_inputs_size, group))

			new_inputs, _ = self.gen_new_inputs(self.all_inputs, group, new_inputs_size, 1, 10)
			new_inputs_overall = new_inputs_overall[:] + new_inputs[:]

			self.app.startsniffer()

			#RUNNING NEW INPUTS
			self.app.run_inputs(new_inputs, self.repetitions_per_input) #Silence the printing of packets

			self.app.sniffer.clean_marked_interactions()
			temp_intrs = self.app.finishexperiment('dim_{}'.format(i+1))
			processed_intrs = [[Utils.convert2packet(p) for p in temp_intr] for temp_intr in temp_intrs]
			all_intrs_overall = all_intrs_overall[:] + processed_intrs[:]
			intrs = self.interactions + processed_intrs

			#TODO Fix this stupid file naming from Utils.process_all, it chops last 5 chars.
			pcap_filename = '{}_run_dim_{}.pcap'.format(self.exp_name, i+1)
			
			print("Results for mutators {}/{}: {}".format(i+1, len(self.grouped_mutators), group))

			#Calling the sniffer with the interactions, returns the leakage which we will not care for now.
			(dim_space_features, dim_time_features, dim_space_tags, dim_time_tags, space_all_leakage, time_all_leakage) = Utils.process_all(ports=self.ports,
				pcap_filename=pcap_filename, interactions=intrs, intervals=None, calcSpace=self.calcSpace, calcTime=self.calcTime, use_phases=True, quant_mode='kde')

			print("Number of NEW interactions: {}".format(len(processed_intrs)))
			print("Number of interactions: {}".format(len(intrs)))

			#print space_all_leakage
			#print time_all_leakage

			if self.calcSpace:
				results[i] = []
				for j,tag in enumerate(related_tags):
					for leakage, t in space_all_leakage:
						if t == tag:
							results[i].append(leakage)
							new_weights[i] = new_weights[i] + abs(leakage-initial_results[j])
							#print tag, abs(leakage-initial_results[j])
			if self.calcTime:
				results[i] = []
				for j, tag in enumerate(related_tags):
					for leakage, t in time_all_leakage:
						if t == tag:
							results[i].append(leakage)
							new_weights[i] = new_weights[i] + abs(leakage-initial_results[j])
							#print tag, abs(leakage-initial_results[j])

		#for i in range(len(results)):
		#	print "Grouped Mutators: {}".format(self.grouped_mutators[i])
		#	print "Results: {}".format(results)
		#	print "Initial Results: {}".format(initial_results)
			#new_weights[i] = abs(results[i][0] - initial_results[0])
			#new_weights[i] = sum([abs(a-b) for (a,b) in zip(results[i][:self.mw_num_features], initial_results[:self.mw_num_features])])

		print("{} New Weights with all features: {}".format(self.exp_name, zip(self.grouped_mutators, new_weights)))

		full_weights = []
		new_mutators = []
		for i, group in enumerate(self.grouped_mutators):
			for el in group:
				new_mutators.append(el)
				full_weights.append(new_weights[i])

		new_weights = full_weights[:]
		if sum(new_weights) == 0:
			new_weights = [1.0/len(new_weights)]*len(new_weights)
		else:
			new_weights = [x/sum(new_weights) for x in new_weights]

		self.mutator_weights = new_weights[:]
		self.mutators = new_mutators[:]
		#self.interactions = self.interactions + all_intrs_overall
		#self.all_inputs = self.all_inputs + new_inputs_overall
		print("{} New Weights: {}".format(self.exp_name, zip(new_mutators,new_weights)))

		return None

	def sample_size_formula(self,pop_mean, pop_stddev, confidence_t=1.96, absolute_error=0.025):
		"""
		Samples per input estimation formula, Cochran, 1977, Ch. 4.6
		t value for 1.96 is 95% confidence, we want relative 5% error on estimated mean.
		"""
		return ((confidence_t*pop_stddev)/(absolute_error))**2

	#TODO Add a new function that increases rpi gradually and looks for new behaviours instead of the sample size formula.
	def find_rpi(self, leakage_sublist):
		def myround(x, base=.05, prec=5):
			return round(base * round(float(x)/base),prec)

		#Initial Variables
		num_inputs_rpi = self.rpi_inputs_to_run
		rpi_eval = self.rpi_eval_value

		rpi_limit = self.rpi_bound

		#num_samples_list = []

		random.shuffle(self.all_inputs)

		num_samples_list = [[0.0 for j in range(num_inputs_rpi)] for i in range(self.rpi_num_features)]

		#PART 2
		for (inp_num,inp) in enumerate(self.all_inputs[:num_inputs_rpi]):
			self.app.startsniffer()
			self.app.run_inputs([inp], rpi_eval)
			self.app.sniffer.clean_marked_interactions()
			temp_intrs = self.app.finishexperiment('rpi_{}'.format(inp_num))
			intrs = [[Utils.convert2packet(p) for p in temp_intr] for temp_intr in temp_intrs]

			pcap_filename = '{}_rpi_input_{}'.format(self.exp_name, inp_num)

			(space_features, time_features, space_tags, time_tags, space_leakage, time_leakage) = Utils.process_all(ports=self.ports, 
				pcap_filename=pcap_filename, interactions=intrs, intervals=None, calcSpace=self.calcSpace, calcTime=self.calcTime, use_phases=True, dry_mode=True)

			for feat_num, (_, tag) in enumerate(leakage_sublist):
				#Get the feature values for that input, calculate moving mean/var and end mean/var. Plot the moving values, 
				feature_values = []
				if self.calcSpace:
					features = space_features
					tags = space_tags
				elif self.calcTime:
					features = time_features
					tags = time_tags
				for (f, t) in zip(features, tags):
					if t == tag:
						feature_values = f

				feature_values = [myround(x, 0.001) for x in feature_values]
				num_unique = len(set(feature_values))
				pop_mean = np.mean(a=feature_values)
				pop_stddev = np.std(a=feature_values, ddof=1)
				num_samples = num_unique
				#num_samples = self.sample_size_formula(pop_mean, pop_stddev, self.rpi_confidence_const, self.rpi_error_bound)
				#print("{} Number of samples for feature {}: {}".format(self.exp_name, tag, num_samples))
				print("{} Number of samples for feature {}: {}".format(self.exp_name, tag, num_unique))
				
				num_samples_list[feat_num][inp_num] = num_samples
				
				"""
				moving_mean = [np.mean(feature_values[:i]) for i in range(2,len(feature_values))]
				moving_var = [np.var(feature_values[:i], ddof=1) for i in range(2,len(feature_values))]

				print len(moving_mean)
				print len(moving_var)

				#Plot 1
				fig, ax = plt.subplots()
				ax.plot(range(len(moving_mean)), moving_mean)

				ax.set_ylabel("Mean value over time")

				fig.savefig(self.app.folder_name + '{}_mean_for_input_{}.png'.format(tag,inp_num),dpi=1200)
				plt.close(fig)

				fig, ax = plt.subplots()
				ax.plot(range(len(moving_var)), moving_var)

				ax.set_ylabel("Variance value over time")

				fig.savefig(self.app.folder_name + '{}_var_for_input_{}.png'.format(tag,inp_num),dpi=1200)
				plt.close(fig)
				"""
		num_samples_per_feature = [np.max(l) for l in num_samples_list]

		max_rpi = int(np.round(max(num_samples_per_feature)))
		avg_rpi = int(np.round(np.average([np.average(l) for l in num_samples_list])))
		print("{} Maximum of all RPI estimations: {}".format(self.exp_name, max_rpi))
		print("{} Average of all RPI estimations: {}".format(self.exp_name, avg_rpi))
		print("{} RPI limit: {}".format(self.exp_name, rpi_limit))

		self.repetitions_per_input = min(max_rpi, rpi_limit)
		if self.repetitions_per_input < 1:
			self.repetitions_per_input = 1

		print("{} Estimated RPI: {}".format(self.exp_name, self.repetitions_per_input))

		return None

