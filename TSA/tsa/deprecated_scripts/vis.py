#!/usr/bin/env python

import sys, os

from tsa import Sniffer, Visualize, Transform, Utils, plt

assert len(sys.argv) >= 3, "usage: vis.py <file> <port> [<port>...]"

pcappath = os.path.abspath(sys.argv[1])
assert pcappath.endswith(".pcap")

tsspath = pcappath[:-5] + ".tss"
ports = map(int, sys.argv[2:])
print("Reading " + pcappath)

s = Sniffer(ports, offline=pcappath)
s.start()
s.join()
interactions = s.interactions
print("Done reading!\n")
print("\n\nVisualizing...")
#Visualize.spacetime_scatter(interactions)
print("\n\n")
print(pcappath)
print(tsspath)
print("\nSay plo() to plot.")
print("Say tss() to export.")
print("Say extract() to export features.")

plo = lambda: Visualize.spacetime_scatter(interactions); plt.show()
tss = lambda: s.export_tss(tsspath)

def extract():
	# Returns the deltas between two consecutive packets as a timing feature.
	timing_features = Utils.extract_time_features(interactions)

	# Returns total size of whole trace and total size of each phase
	# (each one of user's commands is marked as an other phase) and each packet size as a space feature.
	space_features = Utils.extract_space_features(interactions)

	# Note: Usually ML models (like Neural Networks, Naive Bayes, etc.) require a fixed amount of features 
	# and because network traces may have different number of packets, packet level features generally do not work.
	# For the sake of example of tour_planner, we keep this as is because it always has same amount of packets in each run.
	
	# Returning a list where it returns what the secret is in each interaction.
	secrets = Transform.rd_secrets(interactions)

	# Converting to labels for machine learning methods.
	(labels, secret_dict) = Utils.secrets2labels(secrets)
	
	#Extracting the data to a csv file for further processing.
	Utils.data2csv(timing_features, labels, pcappath[:-5] + "_time.csv")
	Utils.data2csv(space_features, labels, pcappath[:-5] + "_space.csv")

	#After this, labels and features can be fed to a machine learning algorithm for supervised learning.
