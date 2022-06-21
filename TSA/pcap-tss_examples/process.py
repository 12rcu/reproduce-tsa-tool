from tsa import Sniffer, Utils
import time
import datetime
import sys, os

def run_example():
	intvfile = "./numberofcities-e4q38-airplan_2-500x005-004-016-000-SkipGetProps-WithDelete-Nico.tss.json"
	pcapfile = "./numberofcities-e4q38-airplan_2-500x005-004-016-000-SkipGetProps-WithDelete.pcap"
	ports = [8080]
	integration_pts_num = 10000
	use_phases = True

	#Reading file using sniffer
	#print ' '
	#print 'Analyzing file: {}'.format(pcapfile)
	#print 'Time: {}'.format(datetime.datetime.now())
	#print '='*60
	#s = Sniffer(ports, offline=pcapfile, showpackets=False)
	#s.start()
	#s.join() #Sniffer runs in a separate thread, we will wait for the parsing of the file to end.
	#time.sleep(0.5)
	#intrs = s.interactions

	interval = Utils.read_interval(intvfile)

	Utils.process_all(ports=ports, pcap_filename=pcapfile, calcSpace=True, calcTime=True, intervals=interval,
		use_phases=use_phases, quant_mode='normal', plot_folder='./plots/', plot=True)

if __name__ == "__main__":
	run_example()

