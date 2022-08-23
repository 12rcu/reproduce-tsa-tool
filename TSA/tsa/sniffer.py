# Needs to be run as root to capture packets
#Py2/Py3 Compatibility
from __future__ import print_function, unicode_literals, absolute_import, division

#General Python packages
import binascii
import threading
import copy
import json
import re
import random
from itertools import chain, groupby
from collections import defaultdict
#Scapy for network packet captures
from scapy.all import sniff, send, IP, UDP, TCP, Raw, wrpcap # sr, sr1, 
import numpy as np

#Matplotlib library for plots
import matplotlib
matplotlib.use('Agg') # Crashes with SSH connections if this isn't set.
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.colors

plot_folder = '/home/burak/VLab/toolbox/bbp/data/benchmark_plots/gaussian/'

# License: http://creativecommons.org/publicdomain/zero/1.0/

colors = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04', '#69d84f',
 	'#894585', '#70b23f','#d4ffff','#65ab7c','#952e8f','#96f97b',
	'#fcfc81','#a5a391','#388004','#4c9085','#5e9b8a','#efb435',
	'#d99b82','#01386a','#25a36f','#59656d','#75fd63','#21fc0d',
	'#0a5f38','#0c06f7','#61de2a','#3778bf','#2242c7','#533cc6',
	'#9bb53c','#05ffa6','#1f6357','#017374','#0cb577','#ff0789',
	'#afa88b','#08787f','#dd85d7','#a6c875','#a7ffb5','#c2b709',
	'#e78ea5','#966ebd','#ccad60','#ac86a8','#947e94','#983fb2',
	'#ff63e9','#b2fba5','#63b365','#8ee53f','#b7e1a1','#ff6f52',
	'#bdf8a3','#d3b683','#fffcc4','#430541','#ffb2d0','#997570',
	'#ad900d','#c48efd','#507b9c','#7d7103','#fffd78','#da467d',
	'#410200','#c9d179','#fffa86','#5684ae','#6b7c85','#6f6c0a',
	'#7e4071','#009337','#d0e429','#fff917','#1d5dec','#054907',
	'#b5ce08','#8fb67b','#c8ffb0','#fdde6c','#ffdf22','#a9be70',
	'#6832e3','#fdb147','#c7ac7d','#fff39a','#850e04','#efc0fe',
	'#40fd14','#b6c406','#9dff00','#3c4142','#f2ab15','#ac4f06',
	'#c4fe82','#2cfa1f','#9a6200','#ca9bf7','#875f42','#3a2efe',
	'#fd8d49','#8b3103','#cba560','#698339','#0cdc73','#b75203',
	'#7f8f4e','#26538d','#63a950','#c87f89','#b1fc99','#ff9a8a',
	'#f6688e','#76fda8','#53fe5c','#4efd54','#a0febf','#7bf2da',
	'#bcf5a6','#ca6b02','#107ab0','#2138ab','#719f91','#fdb915',
	'#fefcaf','#fcf679','#1d0200','#cb6843','#31668a','#247afd',
	'#ffffb6','#90fda9','#86a17d','#fddc5c','#78d1b6','#13bbaf',
	'#fb5ffc','#20f986','#ffe36e','#9d0759','#3a18b1','#c2ff89',
	'#d767ad','#720058','#ffda03','#01c08d','#ac7434','#014600',
	'#9900fa','#02066f','#8e7618','#d1768f','#96b403','#fdff63',
	'#95a3a6','#7f684e','#751973','#089404','#ff6163','#598556',
	'#214761','#3c73a8','#ba9e88','#021bf9','#734a65','#23c48b',
	'#4b57db','#d90166','#015482','#9d0216','#728f02','#ffe5ad',
	'#4e0550','#f9bc08','#ff073a','#c77986','#d6fffe','#fe4b03',
	'#fd5956','#fce166','#8fae22','#e6f2a2','#89a0b0','#7ea07a',
	'#1bfc06','#b9484e','#647d8e','#bffe28','#d725de','#886806',
	'#b2713d','#1f3b4d','#699d4c','#56fca2','#fb5581','#3e82fc',
	'#a0bf16','#d6fffa','#4f738e','#ffb19a','#5c8b15','#54ac68',
	'#cafffb','#b6ffbb','#a75e09','#152eff','#8d5eb7','#5f9e8f',
	'#63f7b4','#606602','#fc86aa','#8c0034','#758000','#ab7e4c',
	'#030764','#fe86a4','#d5174e','#fed0fc','#680018','#fedf08',
	'#fe420f','#6f7c00','#ca0147','#1b2431','#00fbb0','#db5856',
	'#ddd618','#41fdfe','#cf524e','#21c36f','#a90308','#6e1005',
	'#fe828c','#4b6113','#4da409','#beae8a','#0339f8','#a88f59',
	'#5d21d0','#feb209','#4e518b','#964e02','#85a3b2','#ff69af',
	'#c3fbf4','#2afeb7','#005f6a','#0c1793','#ffff81','#fd4659',
	'#f0833a','#f1f33f','#b1d27b','#fc824a','#71aa34','#b7c9e2',
	'#4b0101','#a552e6','#af2f0d','#8b88f8','#9af764','#a6fbb2',
	'#ffc512','#750851','#c14a09','#fe2f4a','#0203e2','#0a437a',
	'#a50055','#ae8b0c','#fd798f','#bfac05','#3eaf76','#c74767',
	'#b29705','#673a3f','#a87dc2','#fafe4b','#c0022f','#0e87cc',
	'#8d8468','#ad03de','#8cff9e','#94ac02','#c4fff7','#fdee73',
	'#33b864','#fff9d0','#758da3','#f504c9','#adf802','#c1c6fc',
	'#35ad6b','#fffd37','#a442a0','#f36196','#c6f808','#f43605',
	'#77a1b5','#8756e4','#889717','#c27e79','#017371','#9f8303',
	'#f7d560','#bdf6fe','#75b84f','#9cbb04','#29465b','#696006',
	'#947706','#fff4f2','#1e9167','#b5c306','#feff7f','#cffdbc',
	'#0add08','#87fd05','#1ef876','#7bfdc7','#bcecac','#bbf90f',
	'#ab9004','#1fb57a','#00555a','#a484ac','#c45508','#3f829d',
	'#548d44','#c95efb','#3ae57f','#016795','#87a922','#f0944d',
	'#5d1451','#25ff29','#d0fe1d','#ffa62b','#01b44c','#ff6cb5',
	'#6b4247','#c7c10c','#b7fffa','#aeff6e','#ec2d01','#76ff7b',
	'#730039','#040348','#df4ec8','#6ecb3c','#8f9805','#5edc1f',
	'#d94ff5','#c8fd3d','#070d0d','#4984b8','#51b73b','#ac7e04',
	'#4e5481','#876e4b','#58bc08','#2fef10','#2dfe54','#0aff02',
	'#9cef43','#18d17b','#35530a','#ef4026','#3c9992','#d0c101',
	'#1805db','#6258c4','#ff964f','#ffab0f','#8f8ce7','#24bca8',
	'#3f012c','#cbf85f','#ff724c','#280137','#b36ff6','#48c072',
	'#bccb7a','#a8415b','#06b1c4','#cd7584','#f1da7a','#ff0490',
	'#805b87','#50a747','#a8a495','#cfff04','#ffff7e','#ff7fa7',
	'#04f489','#fef69e','#cfaf7b','#3b719f','#fdc1c5','#20c073',
	'#9b5fc0','#0f9b8e','#742802','#9db92c','#a4bf20','#cd5909',
	'#ada587','#be013c','#b8ffeb','#dc4d01','#a2653e','#638b27',
	'#419c03','#b1ff65','#9dbcd4','#fdfdfe','#77ab56','#464196',
	'#990147','#befd73','#32bf84','#af6f09','#a0025c','#ffd8b1',
	'#7f4e1e','#bf9b0c','#6ba353','#f075e6','#7bc8f6','#475f94',
	'#f5bf03','#fffeb6','#fffd74','#895b7b','#436bad','#05480d',
	'#c9ae74','#60460f','#98f6b0','#8af1fe','#2ee8bb','#03719c',
	'#02c14d','#b25f03','#2a7e19','#490648','#536267','#5a06ef',
	'#cf0234','#c4a661','#978a84','#1f0954','#03012d','#2bb179',
	'#c3909b','#a66fb5','#770001','#922b05','#7d7f7c','#990f4b',
	'#8f7303','#c83cb9','#fea993','#acbb0d','#c071fe','#ccfd7f',
	'#00022e','#828344','#ffc5cb','#ab1239','#b0054b','#99cc04',
	'#937c00','#019529','#ef1de7','#000435','#42b395','#9d5783',
	'#c8aca9','#c87606','#aa2704','#e4cbff','#fa4224','#0804f9',
	'#5cb200','#76424e','#6c7a0e','#fbdd7e','#2a0134','#044a05',
	'#0d75f8','#fe0002','#cb9d06','#fb7d07','#b9cc81','#edc8ff',
	'#61e160','#8ab8fe','#920a4e','#fe02a2','#9a3001','#65fe08',
	'#befdb7','#b17261','#885f01','#02ccfe','#c1fd95','#836539',
	'#fb2943','#84b701','#b66325','#7f5112','#5fa052','#6dedfd',
	'#0bf9ea','#c760ff','#ffffcb','#f6cefc','#155084','#f5054f',
	'#645403','#7a5901','#a8b504','#3d9973','#000133','#76a973',
	'#2e5a88','#0bf77d','#bd6c48','#ac1db8','#2baf6a','#26f7fd',
	'#aefd6c','#9b8f55','#ffad01','#c69c04','#f4d054','#de9dac',
	'#11875d','#fdb0c0','#b16002','#f7022a','#d5ab09','#86775f',
	'#c69f59','#7a687f','#042e60','#c88d94','#a5fbd5','#fffe71',
	'#6241c7','#fffe40','#d3494e','#985e2b','#a6814c','#ff08e8',
	'#9d7651','#feffca','#98568d','#9e003a','#287c37','#b96902',
	'#ba6873','#ff7855','#94b21c','#c5c9c7','#661aee','#6140ef',
	'#9be5aa','#7b5804','#276ab3','#feb308','#5a86ad','#fec615',
	'#8cfd7e','#6488ea','#056eee','#b27a01','#0ffef9','#fa2a55',
	'#820747','#7a6a4f','#f4320c','#a13905','#6f828a','#a55af4',
	'#ad0afd','#004577','#658d6d','#ca7b80','#005249','#2b5d34',
	'#bff128','#b59410','#2976bb','#014182','#bb3f3f','#fc2647',
	'#a87900','#82cbb2','#667c3e','#658cbb','#749551','#cb7723',
	'#05696b','#ce5dae','#c85a53','#96ae8d','#1fa774','#40a368',
	'#fe46a5','#fe83cc','#94a617','#a88905','#7f5f00','#9e43a2',
	'#062e03','#8a6e45','#cc7a8b','#9e0168','#fdff38','#c0fa8b',
	'#eedc5b','#7ebd01','#3b5b92','#01889f','#3d7afd','#5f34e7',
	'#6d5acf','#748500','#706c11','#3c0008','#cb00f5','#002d04',
	'#b9ff66','#9dc100','#faee66','#7efbb3','#7b002c','#c292a1',
	'#017b92','#fcc006','#657432','#d8863b','#738595','#aa23ff',
	'#08ff08','#9b7a01','#f29e8e','#6fc276','#ff5b00','#fdff52',
	'#866f85','#8ffe09','#d6b4fc','#020035','#703be7','#fd3c06',
	'#eecffe','#510ac9','#4f9153','#9f2305','#728639','#de0c62',
	'#916e99','#ffb16d','#3c4d03','#7f7053','#77926f','#010fcc',
	'#ceaefa','#8f99fb','#c6fcff','#5539cc','#544e03','#017a79',
	'#01f9c6','#c9b003','#929901','#0b5509','#960056','#f97306',
	'#a00498','#2000b1','#94568c','#c2be0e','#748b97','#665fd1',
	'#9c6da5','#c44240','#a24857','#825f87','#c9643b','#90b134',
	'#fffd01','#dfc5fe','#b26400','#7f5e00','#de7e5d','#048243',
	'#ffffd4','#3b638c','#b79400','#84597e','#411900','#7b0323',
	'#04d9ff','#667e2c','#fbeeac','#d7fffe','#4e7496','#874c62',
	'#d5ffff','#826d8c','#ffbacd','#d1ffbd','#448ee4','#05472a',
	'#d5869d','#3d0734','#4a0100','#f8481c','#02590f','#89a203',
	'#e03fd8','#d58a94','#7bb274','#526525','#c94cbe','#db4bda',
	'#9e3623','#b5485d','#735c12','#9c6d57','#028f1e','#b1916e',
	'#49759c','#a0450e','#39ad48','#b66a50','#8cffdb','#a4be5c',
	'#7a9703','#ac9362','#01a049','#d9544d','#fa5ff7','#82cafc',
	'#acfffc','#fcb001','#910951','#fe2c54','#c875c4','#cdc50a',
	'#fd411e','#9a0200','#be6400','#030aa7','#fe019a','#f7879a',
	'#887191','#b00149','#12e193','#fe7b7c','#ff9408','#6a6e09',
	'#8b2e16','#696112','#e17701','#0a481e','#343837','#ffb7ce',
	'#6a79f7','#5d06e9','#3d1c02','#82a67d','#029386','#95d0fc',
	'#be0119','#c9ff27','#373e02','#a9561e','#caa0ff','#ca6641',
	'#02d8e9','#88b378','#980002','#cb0162','#5cac2d','#769958',
	'#a2bffe','#10a674','#06b48b','#af884a','#0b8b87','#ffa756',
	'#a2a415','#154406','#856798','#34013f','#632de9','#0a888a',
	'#6f7632','#d46a7e','#1e488f','#bc13fe','#7ef4cc','#76cd26',
	'#74a662','#80013f','#b1d1fc','#0652ff','#045c5a','#5729ce',
	'#069af3','#ff000d','#f10c45','#5170d7','#acbf69','#6c3461',
	'#5e819d','#601ef9','#b0dd16','#cdfd02','#2c6fbb','#c0737a',
	'#fc5a50','#ffffc2','#7f2b0a','#b04e0f','#a03623','#87ae73',
	'#789b73','#98eff9','#658b38','#5a7d9a','#380835','#fffe7a',
	'#5ca904','#d8dcd6','#a5a502','#d648d7','#047495','#b790d4',
	'#5b7c99','#607c8e','#0b4008','#ed0dd9','#8c000f','#ffff84',
	'#bf9005','#d2bd0a','#ff474c','#0485d1','#ffcfdc','#040273',
	'#a83c09','#90e4c1','#516572','#fac205','#d5b60a','#363737',
	'#4b5d16','#6b8ba4','#80f9ad','#a57e52','#a9f971','#c65102',
	'#e2ca76','#b0ff9d','#9ffeb0','#fdaa48','#fe01b1','#c1f80a',
	'#36013f','#341c02','#b9a281','#8eab12','#9aae07','#02ab2e',
	'#7af9ab','#137e6d','#aaa662','#0343df','#15b01a','#7e1e9c',
	'#610023','#014d4e','#8f1402','#4b006e','#580f41','#8fff9f',
	'#dbb40c','#a2cffe','#c0fb2d','#be03fd','#840000','#d0fefe',
	'#3f9b0b','#01153e','#04d8b2','#c04e01','#0cff0c','#0165fc',
	'#cf6275','#ffd1df','#ceb301','#380282','#aaff32','#53fca1',
	'#8e82fe','#cb416b','#677a04','#ffb07c','#c7fdb5','#ad8150',
	'#ff028d','#000000','#cea2fd','#001146','#0504aa','#e6daa6',
	'#ff796c','#6e750e','#650021','#01ff07','#35063e','#ae7181',
	'#06470c','#13eac9','#00ffff','#e50000','#653700','#ff81c0',
	'#d1b26f','#00035b','#c79fef','#06c2ac','#033500','#9a0eea',
	'#bf77f6','#89fe05','#929591','#75bbfd','#ffff14','#c20078'
	]

random.shuffle(colors)

new_prop_cycle = cycler('color', colors)
plt.rc('axes', prop_cycle=new_prop_cycle)

class Packet:
	"""
		Built-in basic packet class that contains, source, destination, source and destination ports, raw packet load, raw packet size, timestamp and flags.
	"""

	def __init__(self, src, dst, sport, dport, load, size, time, flags):
		self.src = src
		self.dst = dst
		self.sport = sport
		self.dport = dport
		self.load = load
		self.size = size
		self.time = time
		self.flags = flags

	def __str__(self):
		return 'Packet: Src: {}:{}, Dst: {}:{}, Time:{}, Size:{}, Flags:{}'.format(self.src,
			self.sport, self.dst, self.dport, round(self.time, 6), self.size, self.flags)

	def get_size(self):
		return self.size

	def add_size(self, other):
		self.size += other.get_size()


#Contains all other utilities which are repeated within code.
class Utils(object):
	"""
	This object contains all the utility functions such as parsing files, feature extraction and
	numerical computation functions (e.g. calculation of information leakage).
	"""

	@classmethod
	def cleanupX(cls, interactions, mark_str='X'):
		"""
		Takes a list of traces, and removes unwanted traces that are marked as the designated secret (default is 'X') that occur between captures.
		The example of an unwanted trace segment could be an environment setup before each experiment.

		Args:
			interactions: List of traces.
			mark_str: The secret marker that denotes the secret field of the unwanted traces.

		Returns:
			A list of traces that exclude the unwanted traces.
		"""
		secs = Transform.rd_secrets(interactions)
		return [intr for (sec, intr) in zip(secs, interactions) if sec != mark_str]

	@classmethod
	def parsetssfiles(cls, filename,
					  src_filter=None,
					  src_port_filter=None,
					  dst_filter=None,
					  dst_port_filter=None,
					  match_re=None):
		"""
		Parses .tss (Trace separated sequence) files, converts them to interactions (list of traces) and returns the interactions.
		This exists because parsing and processing .pcap files have caused memory issues when they are too big.

		Args:
			filename: File location of the .tss file.
			src_filter: array of source ip to be considered
			src_port_filter: array of source port to be considered
			dst_filter: array of destination ip to be considered
			dst_port_filter: array of destination port to be considered
			match_re: regex to match {source ip}:{source port}->{destination ip}:{destination port} to be considered

		Returns:
			A list of traces.
		"""
		interactions = []
		new_interaction = []
		#started_interaction = False
		phase_counter = 0
		current_secret = ''
		match_re_obj = None
		if match_re:
			match_re_obj = re.compile(match_re)
		with open(filename, 'r') as file:
			for line in file:
				line = line[:-1]
				#print(line)
				#Empty line check
				if len(line) <= 2:
					continue

				#Skipping the label line
				if line[:3] == 'idx':
					continue

				#If it's the secret line, save the current sec
				if line[:6] == 'SECRET':
					div1 = line.split(':')
					if len(div1) > 1:
						if len(new_interaction) != 0:
							interactions.append(new_interaction[:])
						new_interaction = []
						current_secret = div1[-1]
						phase_counter = 0
						prev_idx = -1
						continue
				else:
					div2 = line.split(',')
					if len(div2) == 1:
						continue
					(idx, time, size, src, dst, srcport, dstport, flags) = (int(div2[0]), float(div2[1]), int(div2[2]), div2[3], div2[4], div2[5], div2[6], div2[7])

					# filtering
					if match_re and not match_re_obj.match(f"{src}:{srcport}->{dst}:{dstport}"):
						continue
					if src_filter and not (src in src_filter):
						continue
					if src_port_filter and not (srcport in src_port_filter):
						continue
					if dst_filter and not (dst in dst_filter):
						continue
					if dst_port_filter and not (dstport in dst_port_filter):
						continue

				if phase_counter == 0:
					temp_packet_load = 'INTERACTION_{}'.format(current_secret)
					p = Packet(src, dst, 55555, 55555, temp_packet_load, len(temp_packet_load), time, 'M')
					new_interaction.append(p)
					phase_counter += 1
				if phase_counter != 0 and 'M' in flags:
					load = 'PHASE_num{}'.format(phase_counter) #We don't have the data for the phase names in .tss's, so we enumerate the phase markers instead.
					phase_counter += 1
				else:
					load = ''
				p = Packet(src, dst, srcport, dstport, load, size, time, flags)

				#OPTIMIZATION FOR PROFIT-ML, ALIGNMENT NEEDS THIS
				#Adding dummy packets to fill the gap between tss indices. These packets will be pruned by process_all and will not be processed.
				#if idx-1 != prev_idx:
				#	for _ in range(idx-1-prev_idx):
				#		new_interaction.append(Packet('', '', '', '', '', 0, time, ''))

				prev_idx = idx
				new_interaction.append(p)
		if len(new_interaction) != 0:
			interactions.append(new_interaction)

		interactions[-1][-1].load = 'STOP'
		#new_interaction[-1].load = 'STOP'
			
		return interactions

	@classmethod
	def convert2packet(cls, p):
		"""
		Takes the original scapy Packet object and converts them to our Packet object which only has 8 fields we use.

		Args:
			p: The scapy Packet object.

		Returns:
			The converted Profit Packet object.
		"""
		#Get src, dst, srcport, dstport, load, size and time from the packet structure.
		#Save it in the packets.
		MARKPORT = 55555

		raw = p.getlayer('Raw')
		ip = p.getlayer('IP')
		tcp = p.getlayer('TCP')
		udp = p.getlayer('UDP')
		load = raw.load.decode('latin-1') if raw is not None and 'load' in raw.fields else ''
		size = len(load)
		sport = ''
		dport = ''
		if udp is not None:
			sport = udp.sport if 'sport' in udp.fields else ''
			dport = udp.dport if 'dport' in udp.fields else ''
		if tcp is not None:
			sport = tcp.sport if 'sport' in tcp.fields and sport == '' else ''
			dport = tcp.dport if 'dport' in tcp.fields and dport == '' else ''

		if ip is not None:
			src = ip.src if 'src' in ip.fields else ''
			dst = ip.dst if 'dst' in ip.fields else ''
		else:
			src = ''
			dst = ''

		time = float(p.time)

		if sport == dport == MARKPORT:
			flags = 'M'
		else:
			istcp = p.haslayer('TCP')
			flags = p.sprintf('%TCP.flags%') if istcp else ''

		# We're not keeping the load in memory unless it's a marker because it's encrypted.
		# We only need the size and timing.
		load = load if flags == 'M' or 'INTERACTION' in load or 'STOP' in load or 'PHASE' in load else ''

		p_new = Packet(src, dst, sport, dport, load, size, time, flags)
		#print(p_new)
		return p_new

	@classmethod
	def old_load(cls, p):
		raw = p.getlayer('Raw')
		load = raw.load if raw is not None and 'load' in raw.fields else ''
		return load

	@classmethod
	def load(cls, pkt):
		'''
		Gets the payload content from the packet.

		Args:
			pkt: the Profit Packet object.

		Returns:
			Payload text of pkt.
		'''
		return pkt.load

	@classmethod
	def src(cls, pkt):
		'''
		Gets the source address from the IP layer of the packet.
		Returns empty string if there is no IP layer in the packet.
		'''
		return pkt.src

	@classmethod
	def dst(cls, p):
		'''
		Gets the destination address from the IP layer of the packet.
		Returns empty string if there is no IP layer in the packet.
		'''
		return p.dst

	@classmethod
	def is_marker(cls, p):
		'''
		Returns true if the packet is a marker packet.
		'''
		payload = cls.load(p)
		return (payload[:5] == 'PHASE' or payload[:11] == 'INTERACTION' or payload[:4] == 'STOP')

	@classmethod
	def is_phase_marker(cls, p):
		'''
		Returns true if the packet is a phase marker packet.
		'''
		payload = cls.load(p)
		return payload[:5] == 'PHASE'

	@classmethod
	def is_intr_marker(cls, p):
		'''
		Returns true if the packet is an interaction marker packet.
		'''
		payload = cls.load(p)
		return payload[:11] == 'INTERACTION'

	@classmethod
	def packetsize(cls, p):
		'''
		Returns the packet size of payload.
		'''
		return p.size

	@classmethod
	def equal_markers(cls, p1, p2):
		'''
		Returns true if both packets are the same markers.
		'''
		p1raw = p1.getlayer('Raw')
		p2raw = p2.getlayer('Raw')
		p1load = p1raw.load if p1raw is not None and 'load' in p1raw.fields else ''
		p2load = p2raw.load if p2raw is not None and 'load' in p2raw.fields else ''

		return ((p1load[:5] == p2load[:5] == 'PHASE') or \
			(p1load[:11] == p2load[:11] == 'INTERACTION') or \
			(p1load[:4] == p2load[:4] == 'STOP')) and \
			(p1load == p2load) and (p1.src == p2.src) and \
			(p1.dst == p2.dst)

	@classmethod
	def read_interval(cls, filename):
		'''
		Reads the alignment json file produced by Nicolas' alignment code,
		processes and returns the json object.
		'''
		intervals = None
		with open(filename, 'r') as file:
			intervals = json.load(file)

		#Cleaning up if one interval doesn't have any information.
		intervals = [x for x in intervals if len(x['interval_list']) != 0]
		return intervals

	@classmethod
	def extract_space_features_interval(cls, interactions):
		'''
		Extracts total size of interaction and total size of each phase.

		interactions is the list of list of packets where each list of packets corresponds to one interaction of the system.
		'''

		phase_sum = Transform.rd_aggregate_interval(interactions)
		space_vectors = Transform.rd_space_vectors_interval(interactions)
		space_vectors = [reduce(lambda x, y: x+y, i, []) for i in space_vectors]

		feature_list = list()

		assertion_string = "length of phase_sum:{0}, length of space_vectors:{1}".format(len(phase_sum), len(space_vectors))
		assert len(phase_sum) == len(space_vectors), assertion_string

		for i in range(len(phase_sum)):
			intr_features = phase_sum[i] + space_vectors[i]
			feature_list.append(intr_features)

		return feature_list

	@classmethod
	def extract_space_features(cls, interactions):
		'''
		Extracts total size of interaction and total size of each phase.

		interactions is the list of list of packets where each list of packets corresponds to one interaction of the system.
		'''
		intr_sum = Transform.rd_aggregate(interactions)
		phase_sum = Transform.rd_aggregate_phase(interactions)
		#space_vectors = Transform.rd_space_vectors(interactions)

		feature_list = list()

		assert len(intr_sum) == len(phase_sum)
		#assert len(intr_sum) == len(space_vectors)

		for i in range(len(phase_sum)):
			intr_features = [intr_sum[i]] + phase_sum[i]# + space_vectors[i]
			feature_list.append(intr_features)

		return feature_list

	@classmethod
	def extract_time_features(cls, interactions):
		'''
		Extracts the timing distance between packets and returns it.

		interactions is the list of list of packets where each list of packets corresponds to one interaction of the system.
		'''
		return Transform.rd_time_deltas(interactions)

	@classmethod
	def secrets2labels(cls, secrets):
		'''
		Converts each secret in string to a corresponding number, useful for compatibility with all kinds of statistical tools.

		Returns a tuple in which the first element is a label list corresponding to the given secret list and
		the second element is a dictionary for mapping secrets to labels.

		secrets is a list of strings where each n'th value corresponds to the secret(class for classifiers) of n'th interaction.
		'''
		secret_dict = dict()
		label_list = [None] * len(secrets)
		counter = 0
		for (i,secret) in enumerate(secrets):
			if secret not in secret_dict:
				secret_dict[secret] = counter
				counter = counter + 1
			label_list[i] = secret_dict[secret]

		return label_list, secret_dict

	@classmethod
	def data2csv(cls,features,labels,filename='x.csv'):
		'''
			Puts the features and corresponding labels of multiple interactions to a csv file for records and processing. Returns nothing.

			features is the list of list of features where each n'th list contains features of n'th interaction.
			labels is a list of integers where each n'th value corresponds to the secret(class for classifiers) of n'th interaction.
			filename is file name of csv file to write.
		'''
		assert len(features) == len(labels)
		with open(filename, 'w') as file:
			for (f_list,l) in zip(features,labels):
				file.write(str(l))
				for f in f_list:
					file.write(','+str(f))
				file.write('\n')

#Contains all methods that take a list of interactions and returns modifications
class Transform(object):

	@classmethod
	def __normalize_from_zero(cls, floats):
		if len(floats) > 0:
			# Subtract first one from all
			return [f-floats[0] for f in floats]
		else:
			return floats

	@classmethod
	def rd_extract_features(cls, phase, tag, calcSpace=True, calcTime=True, padding=False, bag_of_words_flag = False):
		#Phase contains a list of packets.

		#Use that to create phase lists for this file, also do N consecutive phase combination to feed this function.
		#This can handle the whole trace as well.
		markerless_phase = cls.tf_prune_markers(phase)

		space_features = []
		time_features = []
		space_tags = []
		time_tags = []

		padding = False

		#Extracting features
		if calcSpace:
			sizes = Transform.rd_space_vectors(markerless_phase) #Packet sizes for each trace

			if len(sizes) == 0:
				return (space_features, time_features, space_tags, time_tags)
			
			stat_sizes = [[float(len(x)), float(sum(x)), min(x) if len(x)!=0 else 0.0, max(x) if len(x)!=0 else 0.0] for x in sizes]

			len_sizes = 0
			if padding:
				#Making trace length the same. 
				#This lets us extract features of packet N if not all traces are of length N or more.
				len_sizes = max([len(x) for x in sizes])
				sizes = [x + [0 for _ in range(len_sizes-len(x))] for x in sizes]
			else:
				#Shortening the traces to extract features of existing packets.
				len_sizes = min([len(x) for x in sizes])
				sizes = [x[:len_sizes] for x in sizes]

			if bag_of_words_flag:
				unique_sizes = set()
				for i, tr in enumerate(sizes):
					unique_sizes = unique_sizes.union(set(tr))

				unique_size_list = list(unique_sizes)
				unique_size_dict = dict()
				for i, size in enumerate(unique_size_list):
					unique_size_dict[size] = i

				for i, tr in enumerate(sizes):
					bag_of_words = [0.0 for _ in range(len(unique_size_list))]
					for el in tr:
						ind = unique_size_dict[el]
						bag_of_words[ind] += 1.0
					sizes[i] = sizes[i] + bag_of_words[:]

			space_features = [x+y for (x,y) in zip(sizes, stat_sizes)]
			space_features = list(map(list,zip(*space_features)))
			space_tags = ["Size of packet {0} in {1}".format(x,tag) for x in range(1,len_sizes+1)]
			if bag_of_words_flag:
				space_tags = space_tags + ["Number of packets with size {0} in {1}".format(x, tag) for x in unique_size_list]
			space_tags = space_tags + ["Number of packets in {0}".format(tag), "Sum of sizes in {0}".format(tag), "Minimum of sizes in {0}".format(tag), "Maximum of sizes in {0}".format(tag)]

		if calcTime:
			deltas = Transform.rd_time_deltas(markerless_phase)
			#print(len(deltas))
			if len(deltas) == 0:
				return (space_features, time_features, space_tags, time_tags)

			stat_deltas = [[sum(x), np.mean(x) if len(x)!=0 else 0, np.std(x) if len(x)!=0 else 0, min(x) if len(x)!=0 else 0, max(x) if len(x)!=0 else 0] for x in deltas]
			#stat_deltas = [[sum(x)] for x in deltas]

			#Left Alignment of Packets
			len_deltas = 0
			if not padding:
				len_deltas = min([len(x) for x in deltas])
				deltas = [x[:len_deltas] for x in deltas]
			else:
				#len_deltas = int(sum([len(x) for x in deltas])/len(deltas))
				len_deltas = max([len(x) for x in deltas])
				deltas = [x + [0.0 for _ in range(len_deltas - len(x))] for x in deltas]

			#Appending per-packet features and per-phase features
			time_features = [x+y for (x,y) in zip(deltas, stat_deltas)]

			#Transposing the lists to have each element list of that spesific feature
			time_features = list(map(list,zip(*time_features)))

			#Naming the features for usage
			#space_tags = space_tags + ["Sum of sizes in {0}".format(tag), "Avg of sizes in {0}".format(tag),
			#"Variance of sizes in {0}".format(tag), "Minimum of sizes in {0}".format(tag), "Maximum of sizes in {0}".format(tag)]

			time_tags = ["Timing delta between packets {0},{1} in {2}".format(x,x+1,tag) for x in range(1,len_deltas+1)]
			#time_tags = time_tags + ["Timing delta between first and last packet in {0}".format(tag)]
			time_tags = time_tags + ["Timing delta between first and last packet in {0}".format(tag), "Avg of deltas in {0}".format(tag),
			"Std.dev. of deltas in {0}".format(tag), "Minimum of deltas in {0}".format(tag), "Maximum of deltas in {0}".format(tag)]

		return (space_features, time_features, space_tags, time_tags)


	@classmethod
	def tf_remove_intrs_intervals(cls, interactions, intervals):
		'''Takes an interaction and an interval list which contains information
			about which interaction to take into account and which time intervals
			are interesting for that interaction compared to others.
			Returns the interaction list with interactions only in intervals file.
		'''
		pruned = [None] * len(intervals)
		for (i,intv_dict) in enumerate(intervals):
			ind = intv_dict['interaction_num']
			intv_list = intv_dict['interval_list']
			if len(intv_list) != 0 and not ind>=len(interactions):
				#print("Ind:", ind, "i:", i, "Len(pruned):", len(pruned), "Len(intrs):", len(interactions))
				pruned[i] = interactions[ind]
		return pruned

	@classmethod
	def tf_split_directions_for_single_address(cls,interactions):
		DEBUG = False
		'''Takes a list of interactions and splits it into multiple lists of
			interactions, each one containing only packets to an address and 
			packets from that address. This is useful when we're observing 
			traffic to a server that changes its IP for every connection.
			In that case, if the experiment is controlled (host is only 
			communicating with the server and nothing else), this function 
			is useful to extract some direction information.
			Returns a list which contains multiple lists of interactions
			each for one direction of network traffic.
		'''
		#interactions = cls.tf_prune_markers(interactions)
		addresses = set()
		#Finding unique addresses.

		for count, i in enumerate(interactions):
			addresses_i = set()
			#i_ = [p for p in i if not Utils.is_marker(p)]
			for p in i:
				src = p.src
				dst = p.dst
				if src == '' or dst == '' or 'M' in p.flags:
					continue
				if src not in addresses_i:
					addresses_i.add(src)
				if dst not in addresses_i:
					addresses_i.add(dst)
			if count == 0:
				addresses = addresses_i
			else:
				addresses = addresses.intersection(addresses_i)

			if DEBUG and (count==0 or count==len(interactions)-1):
				print('TRACE NUMBER: {}'.format(count+1))
				print('LENGTH OF TRACE: {}'.format(len(i)))
				print('ADDRESS SET OF THIS TRACE: {}'.format(addresses_i))
				print('COMMON ADDRESS SET OF ALL TRACES: {}'.format(addresses))
				print('LENGTH OF COMMON ADDRESS SET: {}'.format(len(addresses)))
				print('-------')

		#Splitting the interactions to 2n different interactions,
		# each one covering one way of network packet travel 
		# (from 1st addr, to 1st addr, from 2nd addr, to 2nd addr, ...)
		direction_list = [list() for _ in range(len(addresses)*2)]
		for (i,d) in enumerate(addresses):
			for intr in interactions:
				new_i_from = list()
				new_i_to = list()
				#Should I prune markers in directional interactions
				#intr_ = [p for p in intr if not Utils.is_marker(p)]
				for p in intr:
					if p.flags == 'M':
						continue
					if p.src == d:
						new_i_from.append(p)
					if p.dst == d:
						new_i_to.append(p)
				direction_list[i].append(new_i_from)
				direction_list[i+1].append(new_i_to)

		return addresses, direction_list

	@classmethod
	def tf_split_directions(cls,interactions):
		'''Takes a list of interactions and splits it into multiple lists of
			interactions, each one containing only packets in 1 direction
			according to source and destination IP addresses.
			Returns a list which contains multiple lists of interactions
			each for one direction of network traffic.
		'''
		DEBUG = False
		#interactions = cls.tf_prune_markers(interactions)
		directions = set()
		#Finding unique directions.

		for count, i in enumerate(interactions):
			directions_i = set()
			#i_ = [p for p in i if not Utils.is_marker(p)]
			for p in i:
				src = p.src
				dst = p.dst
				if src == '' or dst == '' or 'M' in p.flags:
					continue
				if (src, dst) not in directions_i:
					directions_i.add((src,dst))
			if count == 0:
				directions = directions_i
			else:
				directions = directions.intersection(directions_i)

			if DEBUG and (count==0 or count==len(interactions)-1):
				print('TRACE NUMBER: {}'.format(count+1))
				print('LENGTH OF TRACE: {}'.format(len(i)))
				print('DIRECTION SET OF THIS TRACE: {}'.format(directions_i))
				print('COMMON DIRECTION SET OF ALL TRACES: {}'.format(directions))
				print('LENGTH OF COMMON DIRECTION SET: {}'.format(len(directions)))
				print('-------')

		#Splitting the interactions to n different interactions,
		# each one covering one way of network packet travel
		direction_list = [list() for _ in range(len(directions))]
		for (i,d) in enumerate(directions):
			for intr in interactions:
				new_i = list()
				#Should I prune markers in directional interactions
				#intr_ = [p for p in intr if not Utils.is_marker(p)]
				for p in intr:
					if (p.src, p.dst) == d:
						new_i.append(p)
				direction_list[i].append(new_i)

		return directions, direction_list

	@classmethod
	def tf_split_intervals(cls,interactions, intervals):
		'''Takes a list of interactions and list of intervals and splits each interaction to
			subinteractions where each subinteraction only contains packets of that interval.
			Returns a list which contain a list of list of list of packets.
		'''
		#Removes the traces that are not in interval file (discarded interactions).
		pruned = cls.tf_remove_intrs_intervals(interactions, intervals)

		#Pkt numbers are intervals and both beginning & end index is closed.
		#Interval [1,2] contains both packets with number 1 and 2.
		interaction_list = list() #[None] * len(pruned)
		for (intr,intv_dict) in zip(pruned,intervals):
			intv_list = intv_dict['interval_list']
			new_intr = [list() for i in range(len(intv_list))]

			for (i,p) in enumerate(intr):
				for (j,intv) in enumerate(intv_list):
					if intv[0] <= i and i <= intv[1]:
						new_intr[j].append(p)
			interaction_list.append(new_intr)

		return interaction_list

	@classmethod
	def tf_align_split_intrs(self, split_intrs):
		'''Takes a list of interactions where each interaction contains n subinteractions (like the result of split_intervals)
			and modifies each subinterval where they have same number of packets.
		'''
		num_intervals = len(split_intrs)
		len_list = [[len(j) for j in i] for i in split_intrs]
		min_list = [min(i) for i in zip(*len_list)]
		for intr in split_intrs:
			for (i,sublist) in enumerate(intr):
				ind = min_list[i]
				if ind == 0:
					intr[i] = []
				else:
					intr[i] = sublist[-ind:]
		return split_intrs

	@classmethod
	def rd_aggregate_interval(cls, interactions):
		'''Takes a list of interactions and a list of intervals
			which contain start and end timestamps for interesting packet groups.
			Returns sum of sizes of all packets in that interval for each interval
			not including markers.
		'''
		#new_intr_list = cls.tf_split_intervals(interactions,intervals)
		intv_aggregation = [[sum([Utils.packetsize(p) for p in intv_list]) for intv_list in intr] for intr in interactions]
		full_aggregation = [sum(intr) for intr in intv_aggregation]
		for (i,aggr_val) in enumerate(full_aggregation):
			intv_aggregation[i].insert(0,aggr_val)

		return intv_aggregation

	@classmethod
	def rd_space_vectors_interval(cls, interactions):
		'''Takes a list of interactions and a list of intervals
			which contain start and end timestamps for interesting packet groups.
			Returns sum of sizes of all packets in that interval for each interval
			not including markers.
		'''
		#new_intr_list = cls.tf_split_intervals(interactions,intervals)
		return [[[Utils.packetsize(p) for p in intv_list] for intv_list in intr] for intr in interactions]

	@classmethod
	def rd_time_deltas_interval(cls, interactions,intervals):
		new_intr_list = cls.tf_split_intervals(interactions,intervals)
		timings = [[[p.time for p in intv_list] for intv_list in intr] for intr in new_intr_list]

		deltas = list()
		for intr in timings:
			intr_deltas = list()
			for intv_list in intr:
				for (curr_t,next_t) in zip(intv_list[:-1], intv_list[1:]):
					intr_deltas.append(next_t - curr_t)
			deltas.append(intr_deltas)

		return deltas

	@classmethod
	def rd_time_deltas(cls, interactions):
		'''
		Extracts the timing distance between packets and returns it.

		interactions is the list of list of packets where each list of packets corresponds to one interaction of the system.
		'''
		timings = cls.rd_time_vectors(interactions)

		deltas = list()
		for sample in timings:

			sample_deltas = list()

			for (p,n) in zip(sample[:-1], sample[1:]):
				sample_deltas.append(n-p)

			deltas.append(sample_deltas)

		return deltas

	@classmethod
	def tf_align(cls, interactions):
		'''Takes an interaction list (which is a list of list of packets) and
			aligns each interaction timewise so they start at the same time,
			returns the aligned interaction. We assume each interaction has same
			phases to ensure consistency.

			interactions is the list of list of packets.
		'''

		# Get the minimum time for alignment.
		min_time = float("inf")
		it = copy.deepcopy(interactions)
		for ia in it:
			for p in ia:
				payload = Utils.load(p)
				if ('INTERACTION' in payload) and (p.time < min_time):
					min_time = p.time
					break

		# Align each interaction to the earliest interaction by offset.
		for ia in it:
			offset = ia[0].time - min_time
			for p in ia:
				p.time = p.time - offset

		# Extracting phase names by looping over a single interaction
		phases = list()
		for p in it[0]:
			payload = Utils.load(p)
			if ('PHASE' in payload):
				phases.append(payload)

		# Capturing phase location for each phase and getting the minimum time
		for phase in phases:
			min_time = float("inf")
			phase_loc = list()

			for ia in it:
				for (i,p) in enumerate(ia):
					payload = Utils.load(p)
					if (phase in payload):
						phase_loc.append(i)
						if (p.time < min_time):
							min_time = p.time
						break
			#print(phase)
			#print(phase_loc)
			#print('===========')

			#Aligning the phase packet and every packet after phase location in the interaction
			for (i, ia) in enumerate(it):
				ind = phase_loc[i]
				offset = ia[ind].time - min_time
				for (j, p) in enumerate(ia):
					if (j >= ind):
						p.time = p.time - offset

		return it

	@classmethod
	def tf_prune_markers(cls, interactions):
		'''Takes an interaction list (which is a list of list of packets),
			removes each marker.

			interactions is the list of list of packets.
			size_lim is the upper limit (inclusive) on the size of packets we remove.
			Default is 0.
			remove_marker denotes whether we remove the markers or not if
			they are within the size limit. Default is False.
		'''

		return [[p for p in ia if not Utils.is_marker(p)] for ia in interactions]

	@classmethod
	def tf_prune_size(cls, interactions, size_lim = 0, remove_marker = False):
		'''Takes an interaction list (which is a list of list of packets),
			removes each packet (except markers otherwise noted) which has size
			in terms of byteslower than or equal to the limit and
			returns the modified interaction.

			interactions is the list of list of packets.
			size_lim is the upper limit (inclusive) on the size of packets we remove.
			Default is 0.
			remove_marker denotes whether we remove the markers or not if
			they are within the size limit. Default is False.
		'''
		#If part doesn't care for the marker and can remove it.
		pruned = list()
		if remove_marker:
			pruned = [[p for p in ia \
			if (Utils.packetsize(p) > size_lim)] \
			for ia in interactions]
		else: #Else part keeps the marker specifically
			pruned = [[p for p in ia \
			if ((Utils.packetsize(p) > size_lim) or Utils.is_marker(p))] \
			for ia in interactions]
		return pruned

	@classmethod
	def tf_prune_dupl(cls, interactions):
		'''Takes an interaction list (which is a list of list of packets),
			removes each packet (except markers otherwise noted) which has size
			in terms of byteslower than or equal to the limit and
			returns the modified interaction.

			interactions is the list of list of packets.
			size_lim is the upper limit (inclusive) on the size of packets we remove.
			Default is 0.
			remove_marker denotes whether we remove the markers or not if
			they are within the size limit. Default is False.
		'''
		#If part doesn't care for the marker and can remove it.
		ret_list = list()
		for ia in interactions:
			intr_list = list()
			previous_packet = None
			for curr_packet in ia:
				if (previous_packet is None) or not(Utils.load(curr_packet) == Utils.load(previous_packet)):
					intr_list.append(curr_packet)
					previous_packet = curr_packet
				else: ##Packet matches with previous packet
					previous_packet = None
			ret_list.append(intr_list)
		return ret_list

	@classmethod
	def rd_aggregate(cls, interactions):
		'''Reduces each interaction to sum of sizes of all packets in the interaction and
			returns the sum of packets and the related secret to that interaction.
			Markers are not counted in the aggregation.
		'''

		#The if part only includes packets that are not markers.
		agg_list = list()

		for ia in interactions:
			#secret = None
			for p in ia:
				payload = Utils.load(p)
				#if ('INTERACTION' in payload):
				#	secret = payload.split('INTERACTION_')[1]

			aggr = sum([Utils.packetsize(p) if (not Utils.is_marker(p)) else 0 for p in ia])
			#Appending secret and aggregated packet size in a tuple
			agg_list.append(aggr)
		return agg_list

	@classmethod
	def rd_aggregate_phase(cls, interactions):
		'''Reduces each phase to sum of sizes of all packets in that phase.
			Markers are not counted in the aggregation.
		'''
		phase_list = list()

		for ia in interactions:
			#secret = None
			publish = False
			counter = 0
			p_list = list()

			for p in ia:
				if (Utils.is_phase_marker(p)):
					if publish:
						p_list.append(counter)
						counter = 0
					publish = True
				elif not Utils.is_marker(p):
					counter = counter + Utils.packetsize(p)
			p_list.append(counter)
			phase_list.append(p_list)

		return phase_list

	#TODO Direction based aggregation.
	"""@classmethod
	def rd_aggregate_phase_dir(cls, interactions):
		'''Reduces each phase to sum of sizes of all packets in that phase. 
			Markers are not counted in the aggregation.
		'''
		phase_list = list()

		for ia in interactions:
			#secret = None
			dir_list = list(set(zip([Utils.src(p) for p in ia],[Utils.dst(p) for p in ia]))) #Getting unique directions for this interaction
			publish = False
			counter_list = [0] * len(dir_list)
			p_list = list()
			
			for p in ia:
				if (Utils.is_phase_marker(p)):
					if publish:
						p_list.append(counter)
						counter_list = [0] * len(dir_list)
					publish = True
				elif not Utils.is_marker(p):

					counter = counter + Utils.packetsize(p)
			p_list.append(counter)
			phase_list.append(p_list)
		
		return phase_list
	"""

	@classmethod
	def rd_aggregate_stats(cls, interactions):
		'''Reduces each phase to sum of sizes of all packets in that phase. (Mean, Variance, Min, Max)
			Markers are not counted in the aggregation.
		'''
		agg_list = cls.rd_aggregate_phase(interactions)
		s = cls.rd_secrets(interactions)
		secrets = reduce(lambda l, x: l.append(x) or l if x not in l else l, s, [])

		ret_list = list()

		for secret in secrets:
			secret_list = list()
			for (s,agg) in agg_list:
				if s == secret:
					secret_list.append(agg)

			a = np.array(secret_list)

			mean     = np.mean(a,axis=0)
			variance = np.std(a,axis=0)
			mins     = np.amin(a,axis=0)
			maxs     = np.amax(a,axis=0)
			zipped   = zip(mean, variance, mins, maxs)
			ret_list.append((secret,zipped))

		return ret_list

	@classmethod
	def rd_space_vectors(cls, interactions):
		'''Reduces each interaction to a vector that contains sizes of each packet.
			Passing interactions through a pruning function like tf_prune() is recommended
			to remove handshake packets. Markers are marked as size -1.

			interactions is the list of list of packets.
		'''
		return [[Utils.packetsize(p) if not Utils.is_marker(p) else -50 for p in ia ] \
			for ia in interactions]

	@classmethod
	def rd_time_vectors(cls, interactions):
		'''Reduces each interaction to a vector that contains times of each packet.
			Handshake packets can be removes by first passing interactions through a pruning
			function like tf_prune_size(). Interactions is the list of list of packets.
		'''
		return [[p.time for p in ia ] \
			for ia in interactions]

	@classmethod
	def rd_time_vectors_normalized(cls, interactions):
		'''Reduces each interaction to a vector that contains the timings of each packet.
			Instead of writing timings for each packet, we can also add first timing as 0
			and rest as time differences according to the first.

			interactions is the list of list of packets.
		'''
		return [cls.__normalize_from_zero([p.time for p in ia]) for ia in interactions]

	@classmethod
	def rd_marker_vectors(cls, interactions):
		'''Reduces each interaction to a vector that denotes whether or not a packet is a marker.
			interactions is the list of list of packets.
		'''
		return [[0 if not Utils.is_marker(p) else 1 for p in ia ] \
			for ia in interactions]

	@classmethod
	def rd_packets_partitioned(cls, interactions):
		rlist = list()
		marks = cls.rd_marker_vectors(interactions)
		for mm,tt in zip(marks,interactions):
			intr_list =list()
			phase_list = list()
			for m,t in zip(mm,tt):
				if (m==1):
					intr_list.append(phase_list)
					phase_list = list()
				else:
					phase_list.append(t)
			del intr_list[0]
			intr_list.append(phase_list)
			rlist.append(intr_list)
		return rlist

	@classmethod
	def rd_time_partitioned(cls, interactions):
		'''Reduces each interaction to a vector that contains the timing info of each packet.
		Then partitions this vector of time information by phase.
		Interactions is the list of list of packets.
		'''
		tp = cls.rd_time_vectors(interactions)
		marks = cls.rd_marker_vectors(interactions)
		rlist = list()
		for mm,tt in zip(marks,tp):
			intr_list = list()
			phase_list = list()
			for m, t in zip(mm,tt):
				if (m == 1):
					intr_list.append(phase_list)
					phase_list = list()
				else:
					phase_list.append(t)
			del intr_list[0]
			intr_list.append(phase_list)
			rlist.append(intr_list)
		return rlist

	@classmethod
	def rd_time_by_phase(cls, interactions):
		'''Reduces each interaction to a vector of the time spent in each phase.'''
		times = cls.rd_time_partitioned(interactions)
		rlist = list()
		for i in times:
			intr_list = list()
			for p in i:
				if len(p) < 2:
					intr_list.append(0)
				else:
					start = p[0]
					end = p[-1]
					intr_list.append(end -start)
			rlist.append(intr_list)
		return rlist

	@classmethod
	def rd_space_partitioned(cls, interactions):
		sp = cls.rd_space_vectors(interactions)
		rlist = list()
		for l in sp:
			intr_list = list()
			phase_list = list()
			for i in l:
				if (i == -50):
					intr_list.append(phase_list)
					phase_list = list()
				else:
					phase_list.append(i)
			del intr_list[0]
			intr_list.append(phase_list)
			rlist.append(intr_list)
		return rlist

	@classmethod
	def rd_secrets(cls, interactions):
		'''Reduces each interaction to the related string, can be used with other methods to
			correlate each interaction with the corresponding secret.

			interactions is the list of list of packets.
		'''
		rlist = list()
		for ia in interactions:
			secret = None
			for p in ia:
				payload = Utils.load(p)
				if ('INTERACTION' in payload):
					secret = payload.split('INTERACTION_')[1]
			rlist.append(secret)

		return rlist

	@classmethod
	def rd_markers(cls, interactions):
		rlist = list()
		for ia in interactions:
			sublist = list()
			for p in ia:
				if (Utils.is_marker(p)):
					sublist.append(Utils.load(p))
			rlist.append(sublist)

		return rlist

#Contains all scapy related sniffer utilities
class Sniffer(threading.Thread):
	'''
	Contains all utilities related to sniffing and recording network packet traces.
	'''

	MARK_PORT = 55555

	def __init__(self, ports, offline=None, showpackets=False):
		threading.Thread.__init__(self)
		self.ports = ports
		self.filterstring = self.makefilterstring(ports)
		self.packets = []
		self.interactions = []
		self.processed_intrs = []
		#self.debug("ports", self.ports)
		#self.debug("filterstring", self.filterstring)
		self.showpackets = showpackets
		#self.showpackets = True
		self._first_time_archive_current_interaction = True
		self.offline = offline
		self.original_pcap = None
		self.pcap = None
		self.last_packet_seen = None
		self.dest = '240.1.1.1'

	def debug(self, label, value=None):
		if value:
			print("[Sniffer]  %s = %s" % (str(label), str(value)))
		else:
			print("[Sniffer]  %s" % str(label))

	def run(self):
		self.debug("Starting")
		# sniff on any interface
		#self.pcap = sniff(filter=self.filterstring,
		self.original_pcap = sniff(lfilter=self.lfilterfunction,
			prn=self.handle_packet,
			stop_filter=self.stop_criterion,
			offline = self.offline)
		if(self.offline is not None):
			self.stop()
		# Creates the interaction file for interaction
		self.pcap = self.cleanup2interaction()
		self.debug("Finished")

	def export(self, name):
		'''
		Exports the saved packet trace to a pcap file location given as an argument.
		'''
		wrpcap(name, self.pcap)

	def cleanup2interaction(self):
		'''

		'''
		new_pcap = list()
		for (curr_packet, next_packet) in zip(self.original_pcap[:-1], self.original_pcap[1:]):
			if not Utils.equal_markers(curr_packet, next_packet):
				new_pcap.append(curr_packet)
		new_pcap.append(self.original_pcap[-1])

		#If there are packets that get sniffed before we start the interaction, we'd like to remove those packets from our experiment.
		first_marker_index = 0
		for (i, pkt) in enumerate(new_pcap):
			pktraw = pkt.getlayer('Raw')
			pktload = pktraw.load.decode('latin_1') if pktraw is not None and 'load' in pktraw.fields else ''
			if pktload.startswith('INTERACTION'):
				first_marker_index = i
				new_pcap = new_pcap[first_marker_index:]
				break

		interactions = list()
		prev_intr_index = 0
		for (i, pkt) in enumerate(new_pcap):
			pktraw = pkt.getlayer('Raw')
			pktload = pktraw.load.decode('latin_1') if pktraw is not None and 'load' in pktraw.fields else ''
			if pktload.startswith('INTERACTION') and i != prev_intr_index:
				interactions.append(new_pcap[prev_intr_index:i])
				prev_intr_index = i

		interactions.append(new_pcap[prev_intr_index:])
		self.interactions = interactions
		self.processed_intrs = [[Utils.convert2packet(p) for p in temp_intr] for temp_intr in self.interactions]

		return new_pcap

	def cleanup2interactiot(self, pktl):
		'''

		'''
		new_pcap = list()
		# only append pkt not same
		for (curr_packet, next_packet) in zip(pktl[:-1], pktl[1:]):
			if not Utils.equal_markers(curr_packet, next_packet):
				new_pcap.append(curr_packet)
		new_pcap.append(pktl[-1])
		return new_pcap

	def is_handshake(self, p):
		if p.haslayer(TCP):
			pkt = binascii.hexlify(bytes(p[TCP].payload))
			return pkt[0:6] != b'170303'
		return False

	@staticmethod
	def is_interaction_pkt(pkt) -> bool:
		pktraw = pkt.getlayer('Raw')
		pktload = pktraw.load.decode('latin_1') if pktraw is not None and 'load' in pktraw.fields else ''
		if pktload.startswith('INTERACTION'):
			return True
		return False

	@staticmethod
	def is_stop_pkt(pkt) -> bool:
		pktraw = pkt.getlayer('Raw')
		pktload = pktraw.load.decode('latin_1') if pktraw is not None and 'load' in pktraw.fields else ''
		if pktload.startswith('STOP'):
			return True
		return False

	def _list_of_packet_pairs_to_session(self, pkts: list) -> list:
		result_dict = defaultdict(list)
		for k, pkt in pkts:
			if 'Ether' in pkt:
				if pkt.haslayer('IP') or pkt.haslayer("IPv6"):
					if pkt.haslayer('TCP'):
						ip_src_fmt = "{IP:%IP.src%}{IPv6:%IPv6.src%}"
						ip_dst_fmt = "{IP:%IP.dst%}{IPv6:%IPv6.dst%}"
						addr_fmt = (ip_src_fmt, ip_dst_fmt)
						fmt = "TCP {}:%r,TCP.sport% > {}:%r,TCP.dport%"
						result_dict[pkt.sprintf(fmt.format(*addr_fmt))].append([k, pkt])
					else:
						result_dict["NOT_TCP"].append([k, pkt])
				else:
					result_dict["NOT_TCP"].append([k, pkt])
			else:
				result_dict["NOT_TCP"].append([k, pkt])
		return [(x, result_dict[x]) for x in result_dict]

	def _sessions_to_packet_pairs(self, sessions: list) -> list:
		result_dict = {}
		for k, session in sessions:
			if k != "NOT_TCP":
				for i, pkt in session:
					# print("SESSION%s" % i)
					# print(self.describe_packet(pkt))
					# print(Utils.convert2packet(pkt).size)
					if k in result_dict:

						result_dict[k][1].add_size(Utils.convert2packet(pkt))
					else:
						new_packet = Utils.convert2packet(pkt)
						result_dict[k] = (i, new_packet)
			else:
				# TLS only uses TCP
				self.debug("SESSION_PARSE", "Found one unrelated UDP/ICMP packet, pruned")
		return [result_dict[x] for x in result_dict]

	@classmethod
	def getcols(cls, pkt:Packet):
		return map(str, (pkt.size, pkt.src, pkt.dst, pkt.sport, pkt.dport, pkt.flags,))

	def export_tss(self, pathname,
				   prune_handshake=False,
				   remain_synfin=True, use_session=True):
		'''
		Export the saved packet trace as multiple time series file. The idea is that a .tss file is
		similar to N .csv files, separated by a blank line. This is for easy
		export of multiple interactions to Mathematica etc. in a single file.
		PEND: We should adopt a more standard format for this!
		PEND: What's the relationship between this and data2csv? Clarify in the API.
		'''
		# Projection function: given a packet, return tuple (composite y-value).
		# We also pass the time of the first packet in the interaction, for normalization.


		columnnames = ['idx', 'time', 'size', 'src', 'dst', 'sport', 'dport', 'flags']

		# Now write out the file:
		with open(pathname, 'w') as tssfile:

			SYN, FIN = 0x02, 0x01

			for interaction in self.interactions:
				# Ignore all zero-payload packets except those with the SYN flag enabled.
				# But keep the ORIGINAL packet indices from the pcap file!
				pairs_kept = [(i, p) for (i, p) in enumerate(interaction) if p.haslayer('IP') and
				    (not (self.is_interaction_pkt(p) or self.is_stop_pkt(p))) and
					(len(Utils.old_load(p)) > 0 and (not self.is_handshake(p) if prune_handshake else True) or
					((p.haslayer('TCP') and (p['TCP'].flags & SYN or p['TCP'].flags & FIN)) if remain_synfin else False))]
				if use_session:
					pairs_kept = self._sessions_to_packet_pairs(self._list_of_packet_pairs_to_session(pairs_kept))
				else:
					pairs_kept = [(x1, Utils.convert2packet(x2)) for (x1, x2) in pairs_kept]

				# Look for the secret (assume it's always in 1st packet for now).
				firstpkt = interaction[0]
				assert firstpkt.sport == firstpkt.dport == self.MARK_PORT
				# self.debug("STOP_ERR", type(firstpkt))
				payload = Utils.old_load(firstpkt).decode("utf-8")
				# self.debug("STOP_ERR", type(payload))
				# self.debug("STOP_ERR", payload[:14])
				if not payload.startswith('INTERACTION_'):
					print("Something weird is going on with the interaction, this one doesn't start with INTERACTION pkt.")
					print("Payload:{}".format(payload))
					continue

				assert payload.startswith('INTERACTION_')
				secret = payload.split('INTERACTION_')[1]

				# Print a special line containing the secret.
				tssfile.write('SECRET:%s\n' % secret)

				# Now print the column names.
				tssfile.write(','.join(columnnames))
				tssfile.write('\n')

				# Now print the data rows.
				# Normalize all times based on interpreting the first packet as time zero.
				t0 = float(interaction[0].time)
				times = [(p.time - t0) for (i, p) in pairs_kept]

				# Ensure time sequence is ordered.
				# NOTE: Temporarily converting this assertion to a warning
				# while hunting down a disappearing-start-of-interaction-packet bug.
				#assert all(times[i] <= times[i+1] for i in range(len(times)-1))
				if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
					print("WARNING! Time sequence is not fully ordered!")

				# Look for streaks of two or more adjacent equal times and make them unique
				# by adding epsilon, 2*epsilon, 3*epsilon, etc. The original numbers only
				# had microsecond precision, and we just normalized them to zero, so we can
				# safely add nanosecond epsilons for streaks of up to 1000 equal elements.
				# In most cases, such streaks only have a few equal elements.
				epsilon = 0.000000001
				times = chain(*[
						[(e + i*epsilon) for (i, e) in enumerate(group)]
						for (t, group) in groupby(times)
					])

				for fixedtime, (originalindex, packet) in zip(times, pairs_kept):
					fixedtimestr = '{:.20f}'.format(fixedtime)
					line = ','.join([str(originalindex), fixedtimestr] + [x for x in self.getcols(packet)])
					tssfile.write(line)
					tssfile.write('\n')
				tssfile.write('\n')

	def makefilterstring(self, ports):
		'''Construct a string that describes the packets we want to capture.'''
		# disjunction of all ports of interest
		disj = ' or '.join([('(port %d)' % p) for p in ports])
		# conjunction to characterize special UDP datagrams (marks)
		mark = '(src port %d and dst port %d)' % (self.MARK_PORT, self.MARK_PORT)
		# now put them together
		filterstring = '(tcp or udp) and (%s or %s)' % (disj, mark)
		return filterstring

	def lfilterfunction(self, p):
		return (p.haslayer('TCP') or p.haslayer('UDP')) \
			and (p.sport in self.ports or p.sport == self.MARK_PORT or p.dport in self.ports or p.dport == self.MARK_PORT)

	def _addmark(self, label):
		'''Mark the beginning of something.'''
		# Send an UDP datagram as a marker.
		# Note that source port = dest port = self.MARK_PORT.
		# This is easy to spot and unlikely to happen naturally.
		# Where do we send it to??
		# For now we do this:
		#if os.environ['HOSTNAME'].startswith('nuc1'):
		#	dest = 'nuc2'
		#else:
		#	dest = 'nuc1'

		# Now send it!
		send(IP(dst=self.dest)/
			UDP(sport=self.MARK_PORT, dport=self.MARK_PORT)/
			Raw(load=label), verbose=False)
		# PEND: Is it OK that we're sending and receiving?
		# PEND: Is it guaranteed that we'll receive THIS packet and not some other one?
		#assert reply.haslayer('UDP'), "expected UDP mark but got: " + str(reply)
		#assert reply['UDP'].payload.load == label
		self.debug("MARK", label)

	def newinteraction(self, name=''):
		'''Start of a new interaction.'''
		intrname = 'INTERACTION_' + name
		self._addmark(label=intrname)

	def newphase(self, name):
		'''Start of a new phase within the current interaction.'''
		phasename = 'PHASE_' + name
		self._addmark(label=phasename)

	def newstatic(self, name):
		'''Start of requesting static file.'''
		staticname = 'STATIC_' + name
		self._addmark(label=staticname)

	def stop(self):
		'''Stop sniffing.'''
		if self.packets:
			self._archive_current_interaction()
		# Send ourselves the stop signal, triggering the stop criterion.
		if self.offline is None:
			self._addmark('STOP')
			self.join()

	def clean_marked_interactions(self, mark='X'):
		rlist = list()
		for ia in self.interactions:
			secret = None
			for p in ia:
				payload = Utils.old_load(p).decode("utf-8")
				if 'INTERACTION' in payload:
					secret = payload.split('INTERACTION_')[1]
					break
			if secret != mark:
				rlist.append(ia)

		self.interactions = rlist

		self.processed_intrs = [[Utils.convert2packet(p) for p in temp_intr] for temp_intr in self.interactions]

		return None
		#secs = Transform.rd_secrets(self.interactions)
		#for x in secs: print(x)
		#self.interactions = [x for (s,x) in zip(secs, self.interactions) if s != mark]

	def _archive_current_interaction(self):
		'''Store the current interaction and reset to empty. (First time, does nothing!)'''
		if self._first_time_archive_current_interaction:
			# First time, do nothing.
			self._first_time_archive_current_interaction = False
		else:
			# Not first time
			self.interactions.append(self.packets[:])
			self.packets = []

	def stop_criterion(self, p):
		'''This gets called on each packet to see if we should stop sniffing.'''
		if p.haslayer('UDP'):
			u = p['UDP']
			# self.debug("STOP_ERR", "%s-%s-%s" % (self.MARK_PORT, u.sport, u.dport))
			if u.sport == self.MARK_PORT and u.dport == self.MARK_PORT:
				assert u.payload, "expected nonempty datagram, but got: " + u.payload
				# self.debug("STOP_ERR", u.payload.load)
				if u.payload.load == b'STOP':
					return True
		return False

	def handle_packet(self, p):
		'''This gets called on every packet.'''
		#if self.last_packet_seen and Utils.equal_markers(p, self.last_packet_seen):
		#	return
		# Update this
		#self.last_packet_seen = p
		# Append to current interaction
		#if('INTERACTION' in Utils.load(p)):
		#	self._archive_current_interaction()
		#self.packets.append(p)
		# Show it
		if self.showpackets:
			self.debug(self.describe_packet(p))

	def describe_packet(self, p):
		'''Returns a string one-liner describing p.'''
		if p.haslayer('Raw') and p.getlayer('Raw').load.decode('latin1').startswith('INTERACTION'):
			line = p.sprintf("%time%   %16s,IP.src%  %16s,IP.dst%  %8s,TCP.sport% %8s,TCP.dport%  %5s,TCP.flags%")
			load = "%9s b" % p['UDP'].payload if p['UDP'].payload else ""
		elif p.haslayer('TCP'):	
			# PEND: Does this change if we use .payload.load instead of .payload?
			line = p.sprintf("%time%   %16s,IP.src%  %16s,IP.dst%  %8s,TCP.sport% %8s,TCP.dport%  %5s,TCP.flags%")
			load = "%9s b" % len(p['TCP'].payload) if p['TCP'].payload else ""
		elif p.haslayer('UDP'):
			line = p.sprintf("%time%   %16s,IP.src%  %16s,IP.dst%  %8s,UDP.sport% %8s,UDP.dport%  %5s,UDP.flags%")
			load = "%9s b" % len(p['UDP'].payload) if p['UDP'].payload else ""
		else:
			line = p.summary()
			load = ""
		return line + load
