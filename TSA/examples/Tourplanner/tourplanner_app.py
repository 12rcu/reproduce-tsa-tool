import time, os
import requests
import random
import collections

import uuid, tempfile, math

import docker

from tsa import Sniffer
from tsa.new_stac import StacPlatform, StacContainer, ChallengeProgramDriver, App

DELETE_UPLOADED_MAP_AFTER_EACH_ITERATION = True

class TourPlannerApp(App):

	def __init__(self, challengeprogramname, topic, ports):
		self.challengeprogramname = challengeprogramname
		self.topic = topic
		self.ports = ports

		self.exptime = time.time()
		self.exptimestamp = self._maketimestamp()
		self.expname = self._makeexpname(self.challengeprogramname, self.topic)
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')
		#print(self._makebanner("EXPERIMENT: " + self.expname, self.params)
		self.serverhost = 'serverNuc'
		self.sniffer = None
		self.run_counter = 0

		self.folder_name = '/data/{}/'.format(self.expname)
		if not os.path.exists(self.folder_name):
			os.makedirs(self.folder_name)

	def launchcontainers(self):
		# Remove any non-registry containers on all hosts
		StacPlatform.cleanuphosts(['clientNuc', 'serverNuc', 'masterNuc'])
		# Launch the server container
		self.server = StacContainer(self.challengeprogramname, self.serverhost)

	def startserver(self):
		# Start the server process inside the server container
		launchcmd = "cd challenge_program && sed -i.bak 's/JETTY_HOST=127.0.0.1/JETTY_HOST=0.0.0.0/' startServer.sh && ./startServer.sh"
		self.server.execbashcmd(launchcmd, detach=True)
		# Wait a bit until the server is ready to use
		time.sleep(20)

	def startsniffer(self):
		print("Starting sniffer")
		self.sniffer = Sniffer(self.ports, showpackets=False)
		self.sniffer.start()

	def stopsniffer(self):
		self.sniffer.stop()

	def export_output(self):
		pass

	def crappyurlencode(self, s):
		'''Crappy version of urlencode that merely replaces spaces with %20s.'''
		return s.replace(' ', '%20')

	def mkurl(self, host, port, citylist):
		'''Build the request URL, given the list of cities to visit.'''
		querystring = '&'.join(['point=' + self.crappyurlencode(city) for city in citylist])
		return ('https://%s:%s/tour?' % (host, str(port))) + querystring

	def run_inputs(self, inputs, num_samples):
		# Don't show warnings when ignoring SSL certificates
		requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

		time_started = time.time()

		# We now iterate over all maps for each sample number
		# (as opposed to sampling the same map many times repeatedly).
		for samplenumber in range(num_samples):
			for i,inp in enumerate(inputs):

				queryurl = self.mkurl(self.serverhost, self.ports[0],inp.places)
				secret = inp.getsecret()

				#print("Citylist #%d of %d: Sample #%d of %d:\nRequesting: %s" % (i, len(inputs), samplenumber, num_samples, queryurl))
				self.sniffer.newinteraction(secret)
				response = requests.get(queryurl, verify=False)
				assert response.ok, 'Expected HTTP OK status!'
				time.sleep(0.025)

	def finishexperiment(self, end_tag=''):
		self.exptimestamp = self._maketimestamp()
		self.expname = self._makeexpname(self.challengeprogramname, "{}-{}".format(self.topic,end_tag))
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')

		self.endtime = time.time()
		self.elapsed = self.endtime - self.exptime
		print("\n\nFinishing experiment %s after %.0f seconds\n" % (self.expname, self.elapsed))
		# save time file
		self.saveoutput('%.2f\n' % self.elapsed, '.time')
		self.sniffer.stop()
		# save pcap file
		print("Saving " + self.pcapfilename)
		self.sniffer.export(self.folder_name + self.pcapfilename)
		# save tss file
		print("Saving " + self.tssfilename)
		self.sniffer.export_tss(self.folder_name + self.tssfilename)

		return self.sniffer.interactions

	def removecontainers(self):
		# Kill and remove the server container
		self.server.killrm()

	def killrmcontainers(self):
		self.removecontainers()

	def postfinishexperiment(self):
		# Do some magic analysis here...
		pass


