from .quantification import Quantification
from .sniffer import Transform
from .sniffer import Packet
import scapy
from scapy.all import sniff, send, sendpfast, IP, UDP, TCP, Raw, wrpcap
import numpy as np
import copy
import random
import time

class Shaper(object):
	#TODO Add class docstring

	@classmethod
	def extract_location(cls, location_str):
		src_dst_str = location_str.split()[-1].split('->')
		src_ip = src_dst_str[0]
		dst_ip = src_dst_str[1]
		return (src_ip, dst_ip)

	@classmethod
	def modify(cls, traces, feature_tag, param = 0.020):
		strategy = None#Map tag and mitigation selected
		# Example ('Packet 1 size src->dst', 'pad', [100,200])
		# Example ('Sum of sizes', 'inject', [5, 10])
		# Example ('...', 'delay', [0.000, 0.020])


		MAX_VAL = 1500
		MTU = MAX_VAL
		tag = feature_tag
		if 'Size of packet' in tag: #Size of packet INDEX in LOCATION
			pkt_ind = int(tag.split()[3]) - 1
			location = ' '.join(tag.split()[5:])
			target_val = 0
			if 'full trace, both directions' in location or 'interval' in location:
				target_val = max([t[pkt_ind].size for t in traces if len(t) > pkt_ind])
				for t in traces:
					if len(t) > pkt_ind:
						t[pkt_ind].size = target_val if (t[pkt_ind].size <= target_val) else random.randint(t[pkt_ind].size, MAX_VAL)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				target_val = 0
				#Finding Max size of Pkt[i] in location
				for t in traces:
					pkt_num_counter = 0
					for p in t:
						if p.src == src_ip and p.dst == dst_ip:
							pkt_num_counter += 1
							if (pkt_num_counter-1) == pkt_ind and p.size > target_val:
								target_val = p.size
								break
							elif (pkt_num_counter-1) == pkt_ind:
								break

				#Applying the modification to all traces
				for t in traces:
					pkt_num_counter = 0
					for p in t:
						if p.src == src_ip and p.dst == dst_ip:
							pkt_num_counter += 1
							if (pkt_num_counter-1) == pkt_ind and p.size < target_val:
								p.size = target_val
								break
			strategy = (tag, 'pad', target_val)
		elif 'Number of packets with size' in tag: #Number of packets with size SIZE in LOCATION
			#TODO Change this, it's wrong, we need to inject packets, not pad them.
			size = int(tag.split()[5])
			location = ' '.join(tag.split()[7:])
			pkt_diff = 0
			if 'full trace, both directions' in location or 'interval' in location:
				max_num_pkts = max([len([p for p in t if p.size == size]) for t in traces])
				min_num_pkts = min([len([p for p in t if p.size == size]) for t in traces])
				pkt_diff = int(2*(max_num_pkts - min_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_loc = random.randint(1, len(t)-1)
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						pkt.size = size
						#pkt.time = t[rand_loc-1].time
						traces[ind].insert(rand_loc, pkt)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				max_num_pkts = max([len([p for p in t if p.size == size and p.src == src_ip and p.dst == dst_ip]) for t in traces])
				min_num_pkts = min([len([p for p in t if p.size == size and p.src == src_ip and p.dst == dst_ip]) for t in traces])
				pkt_diff = int(2*(max_num_pkts - min_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_loc = random.randint(1, len(t)-1)
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						pkt.size = size
						pkt.src = src_ip
						pkt.dst = dst_ip
						#pkt.time = t[rand_loc-1].time
						traces[ind].insert(rand_loc, pkt)
			strategy = (tag, 'insert', [0, pkt_diff])
		elif 'Number of packets in' in tag: #Number of packets in LOCATION
			location = ' '.join(tag.split()[4:])
			pkt_diff = 0
			#Find the max number of packets, find the average per trace
			#Inject packets randomly equal to the difference to equalize the leakage
			if 'full trace, both directions' in location or 'interval' in location:
				max_num_pkts = max([len(t) for t in traces])
				avg_num_pkts = np.mean([len(t) for t in traces])
				pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_loc = random.randint(1, len(t)-1)
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						if pkt.size > 1:
							pkt.size = random.randint(1, pkt.size)
						#pkt.time = t[rand_loc-1].time
						traces[ind].insert(rand_pkt, pkt)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location) #Inject packets with src->dst combination
				max_num_pkts = max([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
				avg_num_pkts = np.mean([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
				pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						if pkt.size > 1:
							pkt.size = random.randint(1, pkt.size)
						pkt.src = src_ip
						pkt.dst = dst_ip
						traces[ind].insert(rand_pkt, pkt)
			strategy = (tag, 'insert', [0, pkt_diff])
		elif 'Minimum of sizes in' in tag: #Minimum of sizes in LOCATION
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location or 'interval' in location:
				max_min_size = max([min([p.size for p in t]) for t in traces])
				for t in traces:
					for p in t:
						if p.size < max_min_size:
							p.size = max_min_size
						if p.size > 1500:
							p.size = 1500
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location) #Inject packets with src->dst combination
				new_traces = [[p.size for p in t if p.src==src_ip and p.dst==dst_ip] for t in traces]
				min_fn = lambda x: min(x) if len(x) > 0 else 0
				max_min_size = max([min_fn(t) for t in new_traces])
				for t in traces:
					for p in t:
						if p.size < max_min_size and p.src == src_ip and p.dst == dst_ip:
							p.size = max_min_size
						if p.size > 1500:
							p.size = 1500
		elif 'Maximum of sizes in' in tag: #Maximum of sizes in LOCATION
			location = ' '.join(tag.split()[4:])
			max_size = 0
			if 'full trace, both directions' in location or 'interval' in location:
				max_size = max([max([p.size for p in t]) for t in traces])
				for ind,t in enumerate(traces):
					rand_loc = random.randint(1, len(t)-1)
					pkt = copy.deepcopy(t[rand_loc])
					pkt.size = max_size
					#pkt.time = t[rand_loc-1].time
					traces[ind].insert(rand_loc, pkt)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location) #Inject packets with src->dst combination
				new_traces = [[p.size for p in t if p.src==src_ip and p.dst==dst_ip] for t in traces]
				max_fn = lambda x: max(x) if len(x) > 0 else 0
				max_size = max([max_fn(t) for t in new_traces])
				for ind,t in enumerate(traces):
					rand_loc = random.randint(1, len(t)-1)
					pkt = copy.deepcopy(t[rand_loc])
					pkt.size = max_size
					pkt.src = src_ip
					pkt.dst = dst_ip
					traces[ind].insert(rand_loc, pkt)
			strategy = (tag, 'insert', 1, max_size)
		elif 'Sum of sizes in' in tag or 'Mean of sizes in' in tag or 'Std.dev. of sizes in' in tag: #Sum of sizes in LOCATION #AGGREGATE
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location:
				pkt_size_padding = None
				if param is None:
					list_sizes = [sum([p.size for p in t]) for t in traces]
					size_difference = np.mean(list_sizes) - min(list_sizes)
					list_num_pkts = [len(t) for t in traces]
					avg_num_pkts = np.mean(list_num_pkts)
					pkt_size_padding = int(size_difference/(2*avg_num_pkts))
					print('Size difference per packet: {}'.format(pkt_size_padding))
				elif param == 'exp':
					for t in traces:
						for p in t:
							if p.size <= 1:
								p.size = 1
							elif p.size == 2:
								p.size = 2
							elif p.size <= 4:
								p.size = 4
							else:
								div_val = np.log2(float(p.size))
								div_val = np.ceil(div_val)
								p.size = int(min(np.power(2, div_val), 1500))
					strategy = (tag, 'pad', 'exp')
				elif param == 'inject':
					max_num_pkts = max([len(t) for t in traces])
					avg_num_pkts = np.mean([len(t) for t in traces])
					pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
					#new_traces = copy.deepcopy(mod_traces)
					for ind,t in enumerate(traces):
						rand_val = random.randint(0, pkt_diff)
						for _ in range(rand_val):
							rand_loc = random.randint(1, len(t)-1)
							rand_pkt = random.randint(1, len(t)-1)
							pkt = copy.deepcopy(t[rand_pkt])
							pkt.size = random.randint(1, MAX_VAL)
							#pkt.time = t[rand_loc-1].time
							traces[ind].insert(rand_loc, pkt)
					strategy = (tag, 'inject', [0, pkt_diff])
				elif param == 'uniform':
					for t in traces:
						for p in t:
							if p.size >= MTU:
								p.size = MTU
							else:
								p.size = random.randint(p.size+1, MTU)
					strategy = (tag, 'pad', 'uniform')
				elif param == 'uniform255':
					for t in traces:
						for p in t:
							p.size += random.randint(1, 255)
							if p.size > MTU: p.size = MTU
					strategy = (tag, 'pad', 'uniform255')
				elif param == 'mice-elephants':
					for t in traces:
						for p in t:
							if p.size <= 100: p.size = 100
							else: p.size = MTU
					strategy = (tag, 'pad', 'mice-elephants')
				elif param == 'linear':
					for t in traces:
						for p in t:
							div_val = int(p.size/128)
							p.size = int(min((div_val+1)*128, MTU))
					strategy = (tag, 'pad', 'linear')	
				elif param == 'mtu':
					for t in traces:
						for p in t:
							p.size = MTU
					strategy = (tag, 'pad', 'mtu')
				elif param == 'p100':
					for t in traces:
						for p in t:
							if p.size <= 100: p.size = 100
							elif p.size <= 200: p.size = 200
							elif p.size <= 300: p.size = 300
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
					strategy = (tag, 'pad', 'p100')
				elif param == 'p500':
					for t in traces:
						for p in t:
							if p.size <= 500: p.size = 500
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
					strategy = (tag, 'pad', 'p500')
				elif param == 'p700':
					for t in traces:
						for p in t:
							if p.size <= 700: p.size = 700
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
					strategy = (tag, 'pad', 'p700')
				elif param == 'p900':
					for t in traces:
						for p in t:
							if p.size <= 900: p.size = 900
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
					strategy = (tag, 'pad', 'p900')
				else:
					pkt_size_padding = int(param)
					if pkt_size_padding > 0:
						for t in traces:
							for p in t:
								p.size = p.size + random.randint(0,pkt_size_padding)
								if p.size > 1500:
									p.size = 1500
					strategy = (tag, 'pad', [0, int(param)])
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				pkt_size_padding = None
				if param is None:
					list_sizes = [sum([p.size for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces]
					size_difference = np.mean(list_sizes) - min(list_sizes)
					list_num_pkts = [len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces]
					avg_num_pkts = np.mean(list_num_pkts)
					pkt_size_padding = int(size_difference/(2*avg_num_pkts))
					print('Size difference per packet: {}'.format(pkt_size_padding))
				elif param == 'exp':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src != src_ip or p.dst != dst_ip:
								continue
							if p.size <= 1:
								p.size = 1
							elif p.size == 2:
								p.size = 2
							elif p.size <= 4:
								p.size = 4
							else:
								div_val = np.log2(float(p.size))
								div_val = np.ceil(div_val)
								p.size = int(min(np.power(2, div_val), 1500))
					
				elif param == 'inject':
					max_num_pkts = max([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
					avg_num_pkts = np.mean([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
					pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
					#new_traces = copy.deepcopy(mod_traces)
					for ind,t in enumerate(traces):
						rand_val = random.randint(0, pkt_diff)
						for _ in range(rand_val):
							rand_loc = random.randint(1, len(t)-1)
							rand_pkt = random.randint(1, len(t)-1)
							pkt = copy.deepcopy(t[rand_pkt])
							pkt.size = random.randint(1, MAX_VAL)
							pkt.src = src_ip
							pkt.dst = dst_ip
							#pkt.time = t[rand_loc-1].time
							traces[ind].insert(rand_loc, pkt)
					strategy = (tag, 'inject', [0, pkt_diff])
				elif param == 'uniform':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size >= MTU:
									p.size = MTU
								else:
									p.size = random.randint(p.size+1, MTU)
				elif param == 'uniform255':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								p.size += random.randint(1, 255)
								if p.size > MTU: p.size = MTU
				elif param == 'mice-elephants':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 100: p.size = 100
								else: p.size = MTU
				elif param == 'linear':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								div_val = int(p.size/128)
								p.size = int(min((div_val+1)*128, MTU))
				elif param == 'mtu':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								p.size = MTU
				elif param == 'p100':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 100: p.size = 100
								elif p.size <= 200: p.size = 200
								elif p.size <= 300: p.size = 300
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				elif param == 'p500':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 500: p.size = 500
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				elif param == 'p700':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 700: p.size = 700
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				elif param == 'p900':
					strategy = (tag, 'pad', param)
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 900: p.size = 900
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				else:
					strategy = (tag, 'pad', [0, int(param)])
					pkt_size_padding = int(param)
					if pkt_size_padding > 0:
						for t in traces:#TODO Add search for modifying total size
							for p in t:
								if p.src == src_ip and p.dst == dst_ip:
									p.size = p.size + random.randint(0,pkt_size_padding)
									if p.size > 1500:
										p.size = 1500
		# elif 'Mean of sizes in' in tag: #Mean of sizes in LOCATION #AGGREGATE
		# 	location = ' '.join(tag.split()[4:])
		# 	if 'full trace, both directions' in location:
		# 		pass
		# 	elif 'full trace, in direction'  in location:
		# 		pass
		# elif 'Std.dev. of sizes in' in tag: #Std.dev. of sizes in LOCATION #AGGREGATE
		# 	location = ' '.join(tag.split()[4:])
		# 	if 'full trace, both directions' in location:
		# 		pass
		# 	elif 'full trace, in direction'  in location:
		# 		pass
		###TIMING SIDE-CHANNELS
		elif 'Timing delta between first and last packet in' in tag or 'Avg of deltas in' in tag or 'Std.dev. of deltas in' in tag or 'Maximum of deltas in' in tag or 'Minimum of deltas in' in tag: #Timing delta between first and last packet in LOCATION
			#location = ' '.join(tag.split()[8:]) #'full trace, both directions' in location or 'full trace, in direction' in location:
			if param == '20ms':
				strategy = (tag, 'delay', param)
				for t in traces:
					len_t = len(t)
					for ind, p in enumerate(t):
						rand_delay = np.random.uniform(0,0.020)
						p.time += rand_delay
						if ind != len_t-1:
							for px in t[ind+1:]:
								px.time += rand_delay
			elif param == '10ms':
				strategy = (tag, 'delay', param)
				for t in traces:
					len_t = len(t)
					for ind, p in enumerate(t):
						rand_delay = np.random.uniform(0,0.010)
						p.time += rand_delay
						if ind != len_t-1:
							for px in t[ind+1:]:
								px.time += rand_delay
			else:
				avg_time_delay = 0.0
				time_delay_count = 0
				for tr in traces:
					average_delay = 0.0
					count = 0
					for p_ind, p in enumerate(tr[:-1]):
						delay = tr[p_ind+1].time - tr[p_ind].time
						average_delay += delay
						count += 1
					average_delay = average_delay/float(count)
					average_delay = average_delay/2.0

					if average_delay > param: #0.400:
						average_delay = param #0.400

					for p_ind, p in enumerate(tr):
						delay = 0
						if p_ind == len(tr) - 1:
							delay = np.abs(np.random.uniform(0.0,average_delay))
							p.time += delay
						else:
							next_p = tr[p_ind]
							if p.src == next_p.dst and p.dst == next_p.src:
								delay = np.abs(np.random.uniform(0.0,average_delay))
								for x, pkt in enumerate(tr[p_ind:]):
									pkt.time += delay + x*np.abs(np.random.uniform(0.0,0.010))
								#Keep the delta, add cumulative noise shifting all packets
							else:
								delay = np.random.uniform(0.0, next_p.time-p.time)
								p.time += delay #np.random.uniform(p.time, next_p.time)
						avg_time_delay += delay
						time_delay_count += 1
				strategy = (tag, 'delay', float(avg_time_delay/time_delay_count))
				print(f'Average delay per packet: {float(avg_time_delay/time_delay_count):.4f}')
			if False:
				src_ip, dst_ip = cls.extract_location(location)
				pass
		elif 'Timing delta between packets' in tag: #Timing delta between packets 1,2 in LOCATION
			location = ' '.join(tag.split()[8:])
			#Similar to size of pkt i, equalize this measure by padding them
			indices = tag.split()[4]
			ind1 = int(indices.split(',')[0])
			ind2 = int(indices.split(',')[1])
			if 'full trace, both directions' in location or 'full trace, in direction' in location:
				for t in traces:
					delay_val = random.uniform(0, 0.100)
					for i, p in enumerate(t):
						if i >= ind2:
							p.time += delay_val
				strategy = (tag, 'delay', 0.100)
			if False:
				src_ip, dst_ip = cls.extract_location(location)
				pass
		# elif 'Minimum of deltas in' in tag: #Minimum of deltas in LOCATION
		# 	location = ' '.join(tag.split()[4:])
		# 	#TODO Similar to minimum sizes, delay packets to equalize this
		# 	if 'full trace, both directions' in location:
		# 		pass
		# 	elif 'full trace, in direction'  in location:
		# 		src_ip, dst_ip = cls.extract_location(location)
		# 		pass
		# elif 'Maximum of deltas in' in tag: #Maximum of deltas in LOCATION
		# 	location = ' '.join(tag.split()[4:])
		# 	#TODO Similar to maximum sizes, delay packets to equalize this
		# 	if 'full trace, both directions' in location:
		# 		pass
		# 	elif 'full trace, in direction'  in location:
		# 		src_ip, dst_ip = cls.extract_location(location)
		# 		pass
		# elif 'Avg of deltas in' in tag: #Avg of deltas in LOCATION
		# 	location = ' '.join(tag.split()[4:])
		# 	if 'full trace, both directions' in location:
		# 		pass
		# 	elif 'full trace, in direction' in location:
		# 		src_ip, dst_ip = cls.extract_location(location)
		# 		pass
		# elif 'Std.dev. of deltas in' in tag: #Std.dev. of deltas in LOCATION
		# 	location = ' '.join(tag.split()[4:])
		# 	if 'full trace, both directions' in location:
		# 		pass
		# 	elif 'full trace, in direction' in location:
		# 		src_ip, dst_ip = cls.extract_location(location)
		# 		pass

		return traces, strategy

	@classmethod
	def targeted_defense(cls, traces, trace_filename, weights):
		#SETTING THE INITIAL PARAMETERS
		calcSpace = True
		calcTime = True
		pre_time = time.time()
		rep_count = 1
		alignment = False
		silence = True
		options_leakage = []
		feature_reduction = None
		#classifier = 'kde-dynamic'
		classifier = 'rf-classifier'
		print('Feature Reduction Method: {}'.format(feature_reduction))

		strategy_list = []

		w_leakage = weights[0] #1.0
		w_overhead = weights[1] #0.1
		w_toverhead = weights[2] #0.0
		min_objective_fn = 1000

		#Pruning packets with size > 1500, don't know how that's possible. #Change this so that either we merge all packets or split all packets, it's inconsistent right now.
		full_traces = []
		for t in traces:
			new_t = []
			for p in t:
				if p.size <= 1500:
					new_t.append(p)
				elif p.size > 1500:
					old_size = p.size
					while old_size > 1500:
						new_p = copy.deepcopy(p)
						old_size = old_size - 1500
						new_p.size = 1500
						new_t.append(new_p)
					new_p = copy.deepcopy(p)
					new_p.size = old_size
					new_t.append(new_p)
			if len(new_t) > 0:
				full_traces.append(new_t)

		print(len(traces))
		print(len(full_traces))
		print('Trace length distribution: {}'.format(set([len(t) for t in traces])))
		print('Number of packets with size > 1500 per trace:', sum([ len([p for p in tr if p.size > 1500]) for tr in traces])/float(len(traces)))
		print('Number of packets with size > 1500 per trace in pruned traces:', sum([ len([p for p in tr if p.size > 1500]) for tr in full_traces])/float(len(full_traces)))
		print('Number of traces with 1 packets per trace:', sum([ 1 for t in traces if len(t) <= 1]))

		test_traces = []
		train_traces = []

		labels = Transform.rd_secrets(full_traces)
		labels_list = list(set(labels))
		numbers_list = list(range(len(labels_list)))
		labels_to_numbers = {k: v for k, v in zip(labels_list, numbers_list)}
		traces_per_label = [[] for _ in labels_list]
		
		print('Set of labels: {}'.format(labels_list))

		for l, tr in zip(labels, full_traces):
			ind = labels_to_numbers[l]
			traces_per_label[ind].append(tr)
		for l in labels_list:
			ind = labels_to_numbers[l]
			list_length = len(traces_per_label[ind])
			train_traces += traces_per_label[ind][:int(list_length/2)]
			test_traces  += traces_per_label[ind][int(list_length/2):]
		print('Number of full traces: {}'.format(len(full_traces)))
		print('Number of train/test traces: {}, {}'.format(len(train_traces), len(test_traces)))
		#Divide traces to labels, send equal to both parts

		import warnings
		warnings.filterwarnings('ignore')

		#INITIAL RUN FOR NO-PADDING
		quant_time = time.time()
		print('Classifier: {}'.format(classifier))
		classifier1 = 'rf-classifier' #'kde-dynamic'
		(labels, features, tags, orig_leakage, feature_importance) = Quantification.process_all(interactions=train_traces, pcap_filename=None, calcSpace=calcSpace, calcTime=calcTime, quant_mode=classifier1, window_size=None, 
		feature_reduction=feature_reduction, num_reduced_features=10, alignment=alignment, new_direction=False, silent=False)

		print('INITIAL QUANT TIME: {:.2f} seconds'.format(time.time()-quant_time))
		options_leakage.append(('No-mitigation', orig_leakage, 0.0, 0.0))
		target_tags = [x[1] for x in feature_importance]

		print('ALL FEATURES: {}'.format(target_tags))

		# if calculate_bounds:
		# 	accuracy_bound = Quantification.calculate_bayes_error(labels, features, tags, target_tags)
		# 	print('Accuracy Bound for Option {}: {:.2f}'.format('No-mitigation', accuracy_bound))

		print('Leakage for Option {}: {:.2f}'.format('No-mitigation', orig_leakage))
		print('Overhead for Option {}: {:.2f}'.format('No-mitigation', 0.0))
		print('Time Overhead for Option {}: {:.2f}'.format('No-mitigation', 0.0))
		print('RANDOM GUESS ACCURACY for {} classes: {:.2f}'.format(len(labels_list), 1.0/len(labels_list)))
		print('='*40)
		print('%'*40)

		min_objective_fn = w_leakage*1.0 + w_overhead*0.0 + w_toverhead*0.0

		total_size_orig = 0
		total_time_orig = 0.0
		for t in train_traces:
			total_time_orig += abs(t[-1].time - t[0].time)
			for p in t:
				total_size_orig += p.size

		#Loop for Mitigation Strategy Synthesis
		non_improvement_count = 0
		non_improvement_limit = -1#50
		old_target_tags = set()
		t_traces = copy.deepcopy(train_traces) #T

		for i in range(len(target_tags)):
			#Step 1: Target top feature, use distribution to find the padding style, distribute it to the packets if aggregate
			#Select top feature
			tag = None
			if w_toverhead < 10:
				for t in target_tags:
					if t not in old_target_tags: #'Mean' not in t and 'Avg' not in t and 'Std' not in t and 'delta' not in t and 
						tag = t
						break
			else: #If time overhead weight is a big value (meaning the user does not want time overhead), we ignore the time mitigation.
				for t in target_tags:
					if t not in old_target_tags and 'delta' not in t: #'Mean' not in t and 'Avg' not in t and 'Std' not in t and 'delta' not in t and 
						tag = t
						break
			if tag is None:
				print('Went over all the tags, early termination!')
				break
			old_target_tags.add(tag)
			
			#Modify Trace
			new_traces_list = []
			new_strategy_list = []
			if 'Sum of sizes in' in tag:
				for param in [None, 50, 100, 150, 200, 250, 'inject', 'exp', 'linear', 'uniform', 'uniform255', 'mice-elephants', 'mtu', 'p100', 'p500', 'p700', 'p900']:
					print('Targeting tag: {}, parameter: {}'.format(tag, param))
					new_traces, strategy = cls.modify(copy.deepcopy(t_traces), tag, param) # T' = modify(T, feature)
					new_traces_list.append(new_traces)
					new_strategy_list.append(strategy)
			elif 'Timing delta between first and last packet in' in tag or 'Avg of deltas in' in tag or 'Std.dev. of deltas in' in tag or 'Maximum of deltas in' in tag or 'Minimum of deltas in' in tag:
				for param in ['10ms', '20ms', 0.010, 0.020, 0.050, 0.100, 0.200, 0.300]:
					print('Targeting tag: {}, parameter: {}'.format(tag, param))
					new_traces, strategy = cls.modify(copy.deepcopy(t_traces), tag, param) # T' = modify(T, feature)
					new_traces_list.append(new_traces)
					new_strategy_list.append(strategy)
			else:
				print('Targeting tag: {}'.format(tag))
				new_traces,strategy = cls.modify(copy.deepcopy(t_traces), tag, None) # T' = modify(T, feature)
				new_traces_list.append(new_traces)
				new_strategy_list.append(strategy)

			definite_strategy = None
			for opt_ind, new_traces in enumerate(new_traces_list):
				#Step 2: Quantify and Check if the modification improves the privacy
				avg_leakage = 0.0
				for rep in range(rep_count):
					(labels, features, tags, leakage, feature_importance) = Quantification.process_all(interactions=new_traces, pcap_filename=None, calcSpace=calcSpace, calcTime=calcTime, quant_mode=classifier, window_size=None, 
					feature_reduction=feature_reduction, num_reduced_features=10, alignment=alignment, new_direction=False, silent=silence)
					avg_leakage += float(leakage)/rep_count


				#target_tags = [x[1] for x in feature_importance]
				#Calculating the Average Overhead against the original traces
				total_size_mod = 0
				total_time_mod = 0.0
				for t in new_traces:
					total_time_mod += abs(t[-1].time - t[0].time)
					for p in t:
						total_size_mod += p.size

				overhead_mod = float(total_size_mod-total_size_orig)/float(total_size_orig)
				t_overhead_mod = float(total_time_mod-total_time_orig)/float(total_time_orig)
				#abs_overhead_mod += float(total_size_mod-total_size_orig)/(len(mod_traces)*float(rep_count))

				if overhead_mod < 0:
					print('OVERHEAD less than 0!')
					overhead_mod = 0.0
				if t_overhead_mod < 0:
					print('TIME OVERHEAD less than 0!')
					t_overhead_mod = 0.0
				objective_fn = w_leakage*avg_leakage + w_overhead*overhead_mod + w_toverhead*t_overhead_mod

				#Save the results and print to screen
				method_name = 'Targeted_Mitigation_Step_{}'.format(i)
				options_leakage.append((method_name, avg_leakage, overhead_mod, t_overhead_mod))
				#target_tags = [x[1] for x in feature_importance]

				#Bound calculation???
				#if calculate_bounds:
				#	accuracy_bound = cls.calculate_bayes_error(labels, features, tags, target_tags)
				#	print('Accuracy Bound for Option {}: {:.2f}'.format(method_name, accuracy_bound))

				print('Total size: {}, Original size: {}'.format(total_size_mod, total_size_orig))
				print('Total time: {}, Original time: {}'.format(total_time_mod, total_time_orig))

				print('Leakage for Option {}: {:.2f}'.format(method_name, avg_leakage))
				print('Overhead for Option {}: {:.2f}'.format(method_name, overhead_mod))
				print('Time Overhead for Option {}: {:.2f}'.format(method_name, t_overhead_mod))
				print('ObjectiveFN for Option {}: {:.2f}'.format(method_name, objective_fn))
				
				if objective_fn < min_objective_fn:
					definite_strategy = new_strategy_list[opt_ind]
					t_traces = copy.deepcopy(new_traces) #T <- T'
					min_objective_fn = objective_fn 
					non_improvement_count = 0
					print('Improving minimization of the Objective Function {}*leakage + {}*space + {}*time'.format(w_leakage, w_overhead, w_toverhead))
				else:
					non_improvement_count += 1
					print('Not improving minimization, count: {}'.format(non_improvement_count))
			if definite_strategy is not None:
				print(f'Adding Mitigation Method {definite_strategy} to the list.')
				strategy_list.append(definite_strategy)
			
			if non_improvement_count >= non_improvement_limit:
				break
		
		#Packet Relaying
		print('Early Termination or Processed all the features!')
		

		print('Mitigation Strategy:', strategy_list)
		return strategy_list

	#TODO: Add this functionality
	# @classmethod
	# def shape_traffic(cls, strategy, test_traces):
	# 	for str_tuple in strategy:
	# 		pass
	# 	pass

	@classmethod
	def relay_packets(cls, t_traces):
		num_pkts_relay = 10#len(t_traces)
		relay_location = '128.111.40.106' # Testing it on lab computer for experiments, real ip addresses in the traces

		#Extracting directions in the trace
		directions_all = set()
		directions_common = set()
		for tr_ind, tr in enumerate(t_traces):
			directions_t = set()
			for p in tr:
				new_dir = (p.src, p.dst)
				if new_dir not in directions_t:
					directions_t.add(new_dir)
			directions_all = directions_all.union(directions_t)
			if tr_ind == 0:
				directions_common = directions_t
			else:
				directions_common = directions_common.intersection(directions_t)
		
		print(f'All directions in the traces: {directions_all}')
		print(f'Common directions in the traces: {directions_common}')
		
		#Simulating real packet transmission
		print(f"Total Number of Traces to Relay: {len(t_traces)}")
		total_relay_time = 0.0
		for tr_ind, tr in enumerate(t_traces[:num_pkts_relay]):
			print("***")
			print(f'Relaying Modified Trace {tr_ind}')
			if len(tr) < 2: continue
			new_tr = [None for _ in tr]
			for p_ind, p in enumerate(tr):#p.dst int(p.dport)
				if p.flags == 'M':
					print(p.load)
					new_tr[p_ind] = IP(dst=relay_location)/UDP(sport=int(p.sport), dport=55555)/Raw(load=p.load)
				else:#Representing padded packets
					new_tr[p_ind] = IP(dst=relay_location)/UDP(sport=int(p.sport), dport=23456)/Raw(load='\x00'*p.size)
				new_tr[p_ind].time = p.time
			relay_start_time = time.time()
			result = send(new_tr, realtime=True)
			relay_time = float(time.time() - relay_start_time)
			actual_traffic_time = float(tr[-1].time-tr[0].time)
			total_relay_time += relay_time
			print(f'Relay Time: {relay_time:.3f} seconds.')
			print(f'Actual Traffic Time: {actual_traffic_time:.3f} seconds.')
			print(f'Total Relay Time: {total_relay_time:.3f} seconds.')
			if actual_traffic_time > 0.0:
				print(f'Time Overhead: {(relay_time-actual_traffic_time)/actual_traffic_time:.3f}')
			else:
				print('Time Overhead: N/A')
			print(f'Number of Packets: {len(tr)}')
			print("===")
		return None
