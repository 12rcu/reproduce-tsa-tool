from __future__ import print_function, unicode_literals, absolute_import, division
from .sniffer import Transform, Utils
import numpy as np

#Matplotlib library for plots
import matplotlib
matplotlib.use('Agg') # Crashes with SSH connections if this isn't set.
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.colors

#Contains all methods that take a list of interactions and draws them for visualization.
class Visualize(object):
	'''
	Contains all methods that take a list of interactions and draws them for visualization.
	'''

	# Some auxiliary functions that don't need self.
	@classmethod
	def __normalize_from_zero(cls, floats):
		if len(floats) > 0:
			# Subtract first one from all
			return [f-floats[0] for f in floats]
		else:
			return floats

	@classmethod
	def __times_of_each(cls, interactions):
		return [cls.__normalize_from_zero([p.time for p in ia]) for ia in interactions]

	@classmethod
	def __sizes_of_each(cls, interactions):
		return [[Utils.packetsize(p) for p in ia] for ia in interactions]

	@classmethod
	def space1(cls, interactions):
		'''Visualizes packets using only space information, as a multi-series bar plot (each series is an interaction).
			Good for contexts where we know that each interaction has the same number
			of packets, and we have relatively few interactions (<10).

			interactions is the list of list of packets to visualize.
		'''
		palette = ['r','b','u','g','y']
		width = 0.9/(len(interactions)-1)
		counter = 0

		plt.figure()
		fig, ax = plt.subplots()

		for intr in interactions:
			#if (len(intact) == 0):
			#	continue
			sizes = [p.len for p in intr]
			indices = np.arange(len(sizes))
			ax.bar(indices + (counter*width), sizes, width, color=palette[counter%len(palette)])
			counter = counter + 1

		#plt.show()

	@classmethod
	def space2(cls, interactions):
		'''Visualizes packets using only space information, as a bar plot (each figure/window is an interaction).

			interactions is the list of list of packets to visualize.
		'''
		space_list = Transform.rd_space_vectors(interactions)
		secret_list = Transform.rd_secrets(interactions)

		for (ia, ib) in zip(space_list,secret_list):
			fg = plt.figure()
			ax = plt.bar(range(len(ia)), ia)

			plt.title("Interaction with secret: " + ib)
			plt.xlabel("Packet #")
			plt.ylabel("Size of packets (negative means marker)")

	@classmethod
	def time(cls, interactions, ydelta=0.5, zscale=10.0):
		'''Visualizes packets using only time information, as a scatterplot where X is time, Y is interaction ID,
			and bubbles denote packets.

			interactions is the list of list of packets to visualize.
			ydelta is the space between rows, i.e., interactions.
			zscale is the multiplier for number of bytes (bubble size).
		'''
		allx, ally, allz = [], [], []
		timesizes_of_each = zip(cls.__times_of_each(interactions), [[40 for _ in ia] for ia in interactions])
		for i, (timelist, sizelist) in enumerate(timesizes_of_each):
			xs = timelist
			ys = [i*ydelta] * len(xs)
			zs = [num_bytes/zscale for num_bytes in sizelist]
			allx.extend(xs)
			ally.extend(ys)
			allz.extend(zs)

		# Rock and roll.
		plt.figure()
		plt.scatter(allx, ally, s=allz)
		#plt.show()

	def cleanup(self, intrs):
		new_intrs = []
		for intr in intrs:
			new_intr = []
			is_reconnect = False
			capture = False
			for p in intr:
				load = Utils.load(p)
				if Utils.is_intr_marker(p) and 'disconnect' in load:
					is_reconnect = True
				elif Utils.is_phase_marker(p) and ((is_reconnect and 'reconnect' in load) or (not is_reconnect and 'connect_deven' in load)):
					capture = True
				elif Utils.is_phase_marker(p):
					capture = False
				if capture or Utils.is_intr_marker(p):
					new_intr.append(p)
			new_intrs.append(new_intr)
		return new_intrs

	@classmethod
	def spacetime_scatter(cls, interactions, ydelta=1, zscale=10.0, marker='o', verts=None):
		'''Visualizes packets using both space and time information, as a scatterplot where X is time, Y is interaction ID,
			and bubble size is space.

			interactions is the list of list of packets to visualize.
			ydelta is the space between rows, i.e., interactions.
			zscale is the multiplier for number of bytes (bubble size).
			marker is the marker for the scatterplot. If marker is None, verts is used.
			verts is the custom marker which is defined by vector primitives.

			PEND: Auto-compute a good zscale from interactions and ydelta.
		'''

		if marker is None and verts is None:
			verts = verts = list(zip([0., 0.], [1., 0.]))

		# Build allx, ally, allz
		# Flatten all interactions into a single list.
		allx, ally, allz = [], [], []
		timesizes_of_each = zip(cls.__times_of_each(interactions), cls.__sizes_of_each(interactions))
		for i, (timelist, sizelist) in enumerate(timesizes_of_each):
			xs = timelist
			ys = [i*ydelta] * len(xs)
			zs = [num_bytes/zscale for num_bytes in sizelist]
			allx.extend(xs)
			ally.extend(ys)
			allz.extend(zs)

		# Rock and roll.
		plt.figure()
		plt.scatter(allx, ally, s=allz, marker=marker, verts=verts)
		#plt.show()

	@classmethod
	def spacetime_subplot(cls, interactions, marker='o--', bar=False):
		'''Visualizes packets using both space and time information, as a series of subplots where X is time, Y is packet size
			and each subplot starting from top to bottom are interactions.

			interactions is the list of list of packets to visualize.
			marker is the marker for the plot.
			bar is the option to change the plot to a barplot.
		'''

		# Flatten all interactions into a single list.
		secret_list = Transform.rd_secrets(interactions)
		timesizes_of_each = zip(cls.__times_of_each(interactions), \
			cls.__sizes_of_each(interactions), secret_list)
		plot_num = len(timesizes_of_each)

		xmin = 0
		xmax = -1
		ymin = 0
		ymax = -1

		width = 0.00001 #Related to barplot width

		plt.figure()
		for i, (timelist, sizelist, secret) in enumerate(timesizes_of_each):
			xs = timelist
			ys = sizelist
			plt.subplot(plot_num, 1, i+1)
			if (bar):
				plt.bar(xs,ys,width)
			else:
				plt.plot(xs,ys,marker)
			plt.title("Interaction with secret: " + secret)
			plt.xlabel("Time of packets relative to first packet")
			plt.ylabel("Size of packets")

			#Finding the max limit in all plots
			(_, temp_xmax) = plt.xlim()
			(_, temp_ymax) = plt.ylim()
			if temp_xmax > xmax:
				xmax = temp_xmax
			if temp_ymax > ymax:
				ymax = temp_ymax

		#Setting up the max limit for all plots
		for i in range(plot_num):
			plt.subplot(plot_num, 1, i+1)
			plt.ylim(ymin, ymax)
			plt.xlim(xmin, xmax)

		# Rock and roll.
		#plt.show()
