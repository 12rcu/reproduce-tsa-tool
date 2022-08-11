# This experiment simulates noise conditions as if the server locations were somewhere else.
# It is to demonstrate that AutoFeed can handle more aggressive noise conditions and still gives meaningful results.
# This only takes the timing noise brought by such condition into account, we did not simulate packet drops or retransmissions.
# The results might be slightly different from the results on the paper because the noise levels are different each time.

from tsa import Utils, Sniffer, Transform, Quantification
import time
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
	bp_2_W_tss, bp_1_W_tss, tourplanner_W]
for l in time_list:
	l = [x for x in l if 'step' in x]
	l.sort()
	intrs = []
	start_time = time.time()
	for (i, el) in enumerate(l):
		calcSpace = False
		calcTime = False
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
		orig_intrs = intrs[:]
		#No noise vs. noise levels if the servers were at US West, US East, Russia 
		noise_list = [(0.0, 0.0), (3.438/1000, 0.083/1000), (74.636/1000, 3.196/1000), (220.518/1000, 2.382/1000)]
		for (avg_val, std_val) in noise_list:
			#print ("Running the experiments with added noise of mean={} and stddev={}".format(avg_val, std_val))
			intrs = addnoise(orig_intrs[:], avg_val, std_val)
			print("Number of traces: {}".format(len(intrs)))
			print(el)

			quant_mode = 'kde-dynamic'
			print ("QUANTIFICATION MODE: {}, calcSpace: {}, calcTime: {}".format(quant_mode, calcSpace, calcTime))
			(labels, features, tags, leakage) = Quantification.process_all(interactions=intrs, use_phases=True, pcap_filename=el, quant_mode=quant_mode, calcSpace=calcSpace, calcTime=calcTime, plot=False, dry_mode = False, debug_print=False)
			print("len_labels_init:{}, len_features_init:{}, f[0]:{}".format(len(labels),len(features), len(features[0])))
			print(leakage)
			
			mid_time = time.time()
			print("Feature Extraction & Quantification Time: {:.2f}".format(mid_time-start_time))
			print("="*80)
