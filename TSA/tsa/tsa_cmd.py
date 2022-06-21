import argparse
import os

#Linking from other files
from tsa import Utils
from tsa import Sniffer
from tsa import Quantification

parser = argparse.ArgumentParser(description='Analyze network traces for information leakages using TSA tool.')

parser.add_argument('filename', dest='filename', type=str,
					help='A trace file location or folder location containing traces.\
					If a .pcap file is passed, the user must pass which ports to examine as well.')
parser.add_argument('-ports', dest='ports', nargs='+', default=[], help='Pass a list of port namesas denoting ports to examine.')
parser.add_argument('--alignment', dest='alignment', action='store_const',
					const=True, default=False,
					help='Set the trace alignment to on.')

parser.add_argument('--analyze-space', dest='calcSpace', action='store_const',
					const=True, default=False,
					help='Set the space feature extraction and analysis to on.')

parser.add_argument('--analyze-time', dest='calcTimee', action='store_const',
					const=True, default=False,
					help='Set the time feature extraction and analysis to on.')


args = parser.parse_args()

ports = [int(x) for x in args.ports] 

traces = []
if '.pcap' in args.filename:
	#Parse the pcap file.
	sniffer = Sniffer(ports, offline=args.filename)
	sniffer.start()
	sniffer.join()
	sniffer.cleanup2interaction()
	#Files to process
	traces = sniffer.processed_intrs
elif '.tss' in args.filename:
	#Parse the tss file.
	traces = Utils.parsetssfiles(filename=args.filename)
else: #Assuming that it's a folder
	file_list = os.listdir(args.filename)
	file_list = [x for x in file_list if '.pcap' in x or '.tss' in x]
	file_list.sort()
	traces = []

	for (i, el) in enumerate(file_list):
		if '.pcap' in el[-5:]:
			sniffer = Sniffer(ports, offline=el, showpackets=False)
			sniffer.start()
			sniffer.join()
			sniffer.cleanup2interaction()
			new_traces = sniffer.processed_intrs
			traces = traces + new_traces
		elif '.tss' in el[-4:]:
			new_traces = Utils.parsetssfiles(filename=args.filename)
			traces = traces + new_traces
		else:
			continue

if len(traces) > 0:
	_ = Quantification.process_all(interactions=traces, alignment=args.alignment,
	quant_mode='kde-dynamic', calcSpace=args.calcSpace, calcTime=args.calcTime)
else:
	print("""
Something went wrong with parsing the trace files. 
Please use '.pcap' or '.tss' files for trace analysis.
""")
