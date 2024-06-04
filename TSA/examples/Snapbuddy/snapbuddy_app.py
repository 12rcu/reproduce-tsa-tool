import time, os
import requests
import random
import collections

import uuid, tempfile, math

import docker

import tsa as sniffer

import Airplan.genmap
from .snapbuddy_client import SnapCaller
from .snapbuddy import SnapbuddyInput

from tsa.new_stac import StacPlatform, StacContainer, ChallengeProgramDriver, App

class SnapbuddyApp(App):

	def __init__(self, challengeprogramname, topic, ports):
		self.challengeprogramname = challengeprogramname
		self.topic = topic
		self.ports = ports
		self.params = {}

		self.serverhost = 'server'
		self.serverport = 8080
		self.username = 'devenmartinez@hotmail.com'
		self.password = 'PS1Ljv4NPs'
		self.userpublickey = '1234567'
		self.ignore_images = True


		self.exptime = time.time()
		self.exptimestamp = self._maketimestamp()
		self.expname = self._makeexpname(self.challengeprogramname, self.topic)
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')
		#print(self._makebanner("EXPERIMENT: " + self.expname, self.params)
		self.serverhost = 'serverNuc'
		self.sniffer = None
		self.run_counter = 0

		self.ith_request = 0
		
		self.folder_name = '/data/{}/'.format(self.expname)
		if not os.path.exists(self.folder_name):
			os.makedirs(self.folder_name)

	def launchcontainers(self):
		# Remove any non-registry containers on all hosts
		if self.serverhost == "serverNuc":
			StacPlatform.cleanuphosts(['clientNuc', 'serverNuc', 'masterNuc'])
		# Launch the server container
		self.imagename = 'localhost:5000/stac_%s:v1.0' % self.challengeprogramname
		self.dockerenv = {'DOCKER_HOST': 'tcp://%s:4243' % self.serverhost}
		self.dockerclient = docker.from_env(version='auto', environment=self.dockerenv)

	def startserver(self):
		launchcmd = "bash -c 'cd challenge_program && ./start.sh'"
		self.container = self.dockerclient.containers.run(self.imagename, launchcmd, network_mode='host', detach=True)
		self.container = self.dockerclient.containers.get(self.container.id)
		#print("Launched container %s\n" % container.short_id)
		time.sleep(16)

		# Start the server process inside the server container
		#self.container.execbashcmd('cd /home/challenge_program && ./startServer.sh', detach=True)
		# Wait a bit until the server is ready to use
		#time.sleep(10)

	def startsniffer(self):
		print("Starting sniffer")
		self.sniffer = sniffer.Sniffer(self.ports, showpackets=False)
		self.sniffer.start()

	def stopsniffer(self):
		self.sniffer.stop()

	def export_output(self):
		pass

	def run_inputs(self, inputs, num_samples):
		for i in range(num_samples):
			for inp in inputs:
				# Reset container every 10 iterations.
				# I think there is a Docker "restart" command for this. PEND: Look into it!
				if self.ith_request == 10:
					# Kill the container
					time.sleep(0.25)
					print("\nKilling container...")
					self.container.kill()
					print("Removing container...")
					self.container.remove()
					print("Killed and removed!\n")
					time.sleep(0.25)
					# Now restart it
					launchcmd = "bash -c 'cd challenge_program && ./start.sh'"
					self.container = self.dockerclient.containers.run(self.imagename, launchcmd, network_mode='host', detach=True)
					self.container = self.dockerclient.containers.get(self.container.id)
					print("Launched container %s\n" % self.container.short_id)
					time.sleep(17)
					# And restart the counter
					self.ith_request = 0

				self.sniffer.newinteraction(inp.getsecret())

				caller = SnapCaller(self.serverhost, self.serverport, self.username, self.password, self.ignore_images)
				response = caller.process(inp.getbssid(), self.userpublickey)

				self.ith_request += 1

	def finishexperiment(self, end_tag=''):
		self.exptimestamp = self._maketimestamp()
		self.expname = self._makeexpname(self.challengeprogramname, "{}-{}".format(self.topic,end_tag))
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')

		self.endtime = time.time()
		self.elapsed = self.endtime - self.exptime
		print("\n\nFinishing experiment %s after %.0f seconds\n" % (self.expname, self.elapsed))
		# save time file
		#self.saveoutput('%.2f\n' % self.elapsed, '.time')
		self.sniffer.stop()
		# save pcap file
		print("Saving " + self.pcapfilename)
		self.sniffer.export(self.folder_name + self.pcapfilename)
		# save tss file
		#print("Saving " + self.tssfilename)
		#self.sniffer.export_tss(self.folder_name + self.tssfilename)
		
		return self.sniffer.interactions

	def removecontainers(self):
		# Kill and remove the server container
		self.container.kill()
		self.container.remove()

	def killrmcontainers(self):
		self.container.kill()
		self.container.remove()

	def postfinishexperiment(self):
		# Do some magic analysis here...
		pass


