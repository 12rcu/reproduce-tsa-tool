from sniffer import Sniffer, Transform, Visualize, Utils, Packet
import cProfile
import pstats

def convert(intrs):
	return [[Utils.convert2packet(p) for p in intr] for intr in intrs]

file_list = [ "airplan_5-airplan_5_test-NoParams-20181130072016.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130062146.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130054236.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130052507.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130051110.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130045759.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130044514.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130043317.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130042132.pcap",
	"airplan_5-airplan_5_test-NoParams-20181130040930.pcap",]

all_intrs = []
for i,fn in enumerate(file_list):
	s = Sniffer(ports=[8443], offline=fn, showpackets=False)
	s.start()
	s.join()
	all_intrs = all_intrs + convert(s.interactions)
	del s
	print("{}th file processed: {}".format(i+1, fn))

cProfile.run('x = Utils.process_all(ports=[8443], pcap_filename="test-profiling.pcap", interactions=all_intrs, calcTime=False)','onehstats_tue_test')
p = pstats.Stats('onehstats_tue_test')
p.strip_dirs().sort_stats('cumulative').print_stats()
