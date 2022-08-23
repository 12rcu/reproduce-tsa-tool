from tsa import Quantification, Utils, Shaper
from tsa import *
from tsa.sniffer import Packet
import glob
import pandas as pd
import os
import sys
import copy
import time

traces = None

run_related, run_gs, run_sa = True, False, False
time_injection_opts = [False] #[True, False]
run_own_dataset = True
run_smartthings_traces = False
run_device_dataset = False
calculate_bounds = False

#classifiers = ['fcnn-classifier', 'knn-classifier', 'rf-classifier'] #rnn-classifier (Not testing RNN right now.)

classifiers = ['rf-classifier'] 

l = ['grpc_stove_server1.tss']#, 'grpc_switch1.tss', 'grpc_stove_device1.tss', 
	# 'grpc_ac_device1.tss', 'grpc_ac_server1.tss', 'grpc_cctv1.tss',
	# 'awsiot_stove_client1.tss', 'awsiot_stove_device1.tss',
	# 'awsiot_ac_client1.tss', 'awsiot_ac_device1.tss', 
	# 'awsiot_lock_device1.tss','awsiot_lock_client1.tss',]
	#'stomp_ac_client1.tss', 'stomp_ac_device1.tss', 'stomp_lock1.tss',
	#'stomp_lock_device1.tss', 'stomp_stove_client1.tss', 'stomp_stove_device1.tss']

weights_list = [[1.0, 0.01, 100000.0], [1.0, 0.1, 100000.0], 
				[1.0, 1.0, 100000.0],
				[0.1, 1.0, 100000.0], [0.01, 1.0, 100000.0],
				]#,[0.0, 1.0, 100000.0], [1.0, 0.0, 100000.0]
				#[1.0, 0.0, 1.0], [1.0, 0.01, 1.0], [1.0, 0.1, 1.0], [1.0, 1.0, 1.0],
				#[0.1, 1.0, 1.0], [0.01, 1.0, 1.0], [0.0, 1.0, 1.0],]

#weights = (1.0, 1.0, 0.0)

cl = classifiers[0]

for weights in weights_list:
	if run_own_dataset:
		print("Testing Event identification on IoT Benchmark Dataset.")
		for fn in l:
			#for time_injection in time_injection_opts:
			#print("RUNNING EXPS w/ TIME DELAY INJECTION = {}".format(time_injection))
			filename = "./protocol_benchmarks/" + fn
			traces1 = Utils.parsetssfiles(filename)
			traces2 = Utils.parsetssfiles(filename[:-5] + '2.tss')
			traces = traces1[1:] + traces2[1:]
			print("")
			print("")
			print("*"*50)
			print("Analyzing {}, number of traces: {}".format(fn, len(traces)))
			print("*"*50)
			
			x = Shaper.targeted_defense(traces, filename, weights)
		#print("TOTAL ELAPSED TIME: {:.3f} seconds".format(time.time() - very_start_time))

	if run_smartthings_traces:
		print("Testing Event identification on IoT Real World Dataset.")
		folder = "./smartthings_benchmarks/"
		for ind, fn in enumerate(os.listdir(folder)):
			print("Running File {}, Filename: {}".format(ind, folder+str(fn)))
			filename = folder + str(fn)
			traces = Utils.parsetssfiles(filename)
			print("")
			print("")
			print("*"*50)
			print("Analyzing {}, number of traces: {}".format(fn, len(traces)))
			print("*"*50)
			x = Shaper.targeted_defense(traces, filename, weights)
			#print("TOTAL ELAPSED TIME: {:.3f} seconds".format(time.time() - very_start_time))

	if run_device_dataset:
		print("Testing Device identification on combined UNSW dataset.")
		#The folder containing the datasets.
		folder = "./Pinheiro_device_benchmark_traces/"
		alignment = False

		#Maps the MAC address of an IoT device to a label (integer).
		devices ={"d0:52:a8:00:67:5e":1,"44:65:0d:56:cc:d3":2,"70:ee:50:18:34:43":3,"f4:f2:6d:93:51:f1":4,"00:16:6c:ab:6b:88":5,"30:8c:fb:2f:e4:b2":6,"00:62:6e:51:27:2e":7,"00:24:e4:11:18:a8":8,"ec:1a:59:79:f4:89":9,"50:c7:bf:00:56:39":10,"74:c6:3b:29:d7:1d":11,"ec:1a:59:83:28:11":12,"18:b4:30:25:be:e4":13,"70:ee:50:03:b8:ac":14,"00:24:e4:1b:6f:96":15,"74:6a:89:00:2e:25":16,"00:24:e4:20:28:c6":17,"d0:73:d5:01:83:08":18,"18:b7:9e:02:20:44":19,"e0:76:d0:33:bb:85":20,"70:5a:0f:e4:9b:c0":21}

		full_traces = []
		for ind, f in enumerate(os.listdir(folder)):
			print("Running File {}, Filename: {}".format(ind, folder+f))
			traces = []
			if ind >= 1:
				continue
			df = pd.read_csv(folder+str(f))[["eth.src", "TIME", "Size", "IP.src", "IP.dst", "port.src", "port.dst"]]

			#Replaces the MAC address of IoT devices with labels.
			for d in devices:
				df["eth.src"] = df["eth.src"].replace(d,devices[str(d)])

			#Extracts IoT devices from the original dataset.
			df = df[df['eth.src'].astype(str).str.isdigit()]

			#Groups packets into one-second windows for each IoT device.
			g = df.groupby(by=["TIME","eth.src"])

			for (el1, el2) in g:
				interaction = []
				p_list = el2.values
				sec = p_list[0][0]
				src = p_list[0][3]
				dst = p_list[0][4]
				packet_load = 'INTERACTION_{}'.format(sec)
				p = Packet(src, dst, 55555, 55555, packet_load, len(packet_load), 0.0, 'M')
				interaction.append(p)
				for el in p_list:
					#sec   = el[0]
					time_  = el[1]
					size  = el[2]
					src   = el[3]
					dst   = el[4]
					sport = el[5]
					dport = el[6]
					p = Packet(src, dst, sport, dport, '', size, 0.0, '') #src, dst, sport, dport, load, size, time, flags)
					interaction.append(p)
				#print(len(interaction))
				traces.append(interaction)

			print("Len Traces", len(traces))
			print("*"*50)
			print("Analyzing {}, number of traces: {}".format(str(f), len(traces)))
			print("*"*50)
			full_traces = full_traces + traces
		print("Length of COMBINED TRACES: {}".format(len(full_traces)))
		x = Shaper.targeted_defense(traces, filename, weights)

#print("TOTAL RUNTIME: {:.3f} seconds".format(time.time() - very_start_time))
print("END")
