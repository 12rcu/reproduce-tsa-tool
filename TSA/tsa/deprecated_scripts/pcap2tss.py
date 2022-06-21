#!/usr/bin/env python

import sys
import os
import time

from tsa import Sniffer

assert len(sys.argv) >= 3, "usage: pcap2tss.py <file> <port> [<port>...]"

pcappath = os.path.abspath(sys.argv[1])
ports = map(int, sys.argv[2:])
assert pcappath.endswith(".pcap")

tsspath = pcappath[:-5] + ".tss.txt"
noduppath = pcappath[:-5] + ".nodup.pcap"

t0 = time.time()
print("Reading ...")
print(pcappath)
print("\n\n")

s = Sniffer(ports, offline=pcappath)
s.start()
s.join()
print("\n\n")

t1 = time.time()
print("Done loading pcap file, %.2f s" % (t1-t0))
print("\n")

print("len(s.pcap) == %d" % len(s.pcap))
t0 = time.time()
print("Cleaning up ...")
t1 = time.time()
print("Done, %.2f s" % (t1-t0))
# Burak's cleanup function. Split to s.interactions, and also overwrite s.pcap
s.pcap = s.cleanup2interaction()

print("len(s.pcap) == %d" % len(s.pcap))
print("len(s.interactions) == %d" % len(s.interactions))
print("sum(map(len, s.interactions)) == %d" % sum(map(len, s.interactions)))
print("\n")
print(pcappath)
print("\n")

t0 = time.time()
print("Exporting multiple time series...")
print(tsspath)
s.export_tss(tsspath)
t1 = time.time()
print("Done, %.2f s" % (t1-t0))

t0 = time.time()
print("Exporting nodup pcap...")
print(noduppath)
s.export(noduppath)
t1 = time.time()
print("Done, %.2f s" % (t1-t0))

print("Finished!")
