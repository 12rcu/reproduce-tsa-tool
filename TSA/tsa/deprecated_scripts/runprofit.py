#!/bin/env python

from sniffer import Sniffer, Transform, Visualize, Utils
import sys

assert len(sys.argv) == 2, "Need pcap filename"
pathname = sys.argv[1]

s = Sniffer([8443], pathname)
s.run()

x = Utils.process_all([8443], pcap_filename='dummy.pcap', interactions=s.processed_intrs, intervals=None, show_packets=False, calcTime=False)


