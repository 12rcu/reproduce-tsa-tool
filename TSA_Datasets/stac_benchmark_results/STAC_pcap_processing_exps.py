# This experiment processes the AutoFeed pcaps and prints the leakage results at each step. 
# It also prints the stopping criterion if the execution should stop, stopping criterion code copied from AutoProfit class.


from tsa import Utils, Sniffer, Transform, Quantification
import time

import random
import numpy as np
import os

snapbuddy = os.listdir("./snapbuddy/") #*.pcap
gabfeed_1_W = os.listdir("./gabfeed_1/") #*step*.pcap
gabfeed_2_W = os.listdir("./gabfeed_2/") #*step*.pcap
gabfeed_5_W = os.listdir("./gabfeed_5/") #*step*.pcap

airplan_2_W_tss = os.listdir("./airplan_2/") #*step*.tss
airplan_3_W_tss = os.listdir("./airplan_3/") #*step*.tss
airplan_5_W_tss = os.listdir("./airplan_5/") #*step*.tss
bp_1_W_tss = os.listdir("./bidpal_1/") #*.tss
bp_2_W_tss = os.listdir("./bidpal_2/") #*.tss

pb_1_W_tss = os.listdir("./powerbroker_1/") #*.tss
pb_2_W_tss = os.listdir("./powerbroker_2/") #*.tss
pb_4_W_tss = os.listdir("./powerbroker_4/") #*.tss
tourplanner_W = os.listdir("./tour_planner/") #*step*.tss
railyard_W = os.listdir("./railyard/") #*step*.tss 

stop_criterion_length = 3
stop_criterion_num_features = 1
stop_criterion_bound = 0.005 #0.5/100.0

#Adds time noise to each packet to simulate different conditions, assumes the noise is normal
def addnoise(traces, avg_val, stddev_val):
	if stddev_val == 0 and avg_val == 0: return traces
	new_traces = traces[:]
	for t in new_traces:
		additive_noise = float(0.0)
		for p in t:
			additive_noise += float(np.random.normal(avg_val, stddev_val, size=None))
			p.time += additive_noise
	return new_traces

#Stop points selected from experiments in Table 2.
time_list = [gabfeed_1_W, gabfeed_2_W, gabfeed_5_W, 
	pb_1_W_tss, pb_2_W_tss, pb_4_W_tss, 
	bp_2_W_tss, bp_1_W_tss, tourplanner_W, 
	airplan_2_W_tss, airplan_3_W_tss, airplan_5_W_tss, railyard_W, snapbuddy]
for l in time_list:
	l = [x for x in l if 'step' in x]
	l.sort()
	intrs = []
	start_time = time.time()

	max_leakage_history = []
	leakage_history = []

	for (i, el) in enumerate(l):
		calcSpace = False; calcTime = False
		if 'airplan' in el or 'snapbuddy' in el or 'railyard' in el:
			calcSpace = True
		else:
			calcTime = True
		if '.tss' in el:
			new_intrs = Utils.parsetssfiles(el)
		else:
			ports = [8080]
			if 'gabfeed' in el or 'snapbuddy' in el:
				ports = [8080]
			elif 'airplan' in el:
				ports = [8443]
			elif 'railyard' in el:
				ports = [3456]
			elif 'powerbroker' in el:
				ports = [9000, 9001]
			elif 'bidpal' in el:
				ports = [8000, 8001]
			elif 'tourplanner' in el:
				ports = [8989]
			print(ports)
			s = Sniffer(ports, offline=el, showpackets=False)
			s.start()
			s.join()
			s.cleanup2interaction()
			new_intrs = s.processed_intrs
		print("Number of traces added: {}".format(len(new_intrs)))
		print("Number of packets for first element: {}".format(len(new_intrs[0])))
		intrs = intrs + new_intrs
		print("Total number of traces: {}".format(len(intrs)))
		print (i, len(l))
		
		quant_mode = 'kde-dynamic'
		print ("QUANTIFICATION MODE: {}, calcSpace: {}, calcTime: {}".format(quant_mode, calcSpace, calcTime))
		(labels, features, tags, leakage) = Quantification.process_all(interactions=intrs, use_phases=True, pcap_filename=el, quant_mode=quant_mode, calcSpace=calcSpace, calcTime=calcTime, plot=False, dry_mode = False, debug_print=False)
		print("len_labels_init:{}, len_features_init:{}, f[0]:{}".format(len(labels),len(features), len(features[0])))
		print(leakage)
		
		mid_time = time.time()
		print("Feature Extraction & Quantification Time: {:.2f}".format(mid_time-start_time))
		print("="*80)

		secret_leakage = float(np.log2(len(set(Transform.rd_secrets(intrs))))) 
		#Calculate total possible information leakage from number of secrets, take np.log2 of it.
		max_leakage_history.append(secret_leakage)

		print("Leakage for step {}: {}".format(i, leakage))
		leakage_history.append(leakage)
		print('Leakage history in bits: {}'.format(leakage_history))
		lp = [[(leak for (leak, tag) in lx[:10]] for lx in leakage_history]
		print("Leakage in list form:"+ str(lp))

		#STOP CRITERION
		if len(leakage_history) >= stop_criterion_length:
			lp_lastfive = [[float(leakage)/max_leakage_history[-1] for (leakage, tag) in l[:stop_criterion_num_features]] for l in leakage_history[-stop_criterion_length:]]
			stop_criterion = True

			tp_lastfive = map(list, zip(*lp_lastfive))
			for lx in tp_lastfive:
				difference = max(lx) - min(lx) 
				if difference > stop_criterion_bound:
					print("Difference of leakage {:.4f} greater than the bound of {:.4f}".format(difference, stop_criterion_bound))
					stop_criterion = False
					break
				else:
					print("Difference of leakage {:.4f} less than the bound of {:.4f}".format(difference, stop_criterion_bound))
			if stop_criterion:
				print("STOPPING because top {} features did not have any change within the bound of {} for last {} steps.".format(stop_criterion_num_features, stop_criterion_bound, stop_criterion_length))
				#return None
