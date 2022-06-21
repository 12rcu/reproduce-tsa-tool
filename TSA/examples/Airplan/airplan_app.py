import time
import os
import requests

from tsa import Sniffer

import genmap
from airplan_client import AirPlanClient

from tsa.new_stac import StacPlatform, StacContainer, ChallengeProgramDriver, App

DELETE_UPLOADED_MAP_AFTER_EACH_ITERATION = True

class AirplanApp(App):

	def __init__(self, challengeprogramname, topic, ports):
		self.challengeprogramname = challengeprogramname
		self.topic = topic
		self.ports = ports
		self.params = {}

		self.exptime = time.time()
		self.exptimestamp = self._maketimestamp()
		self.expname = self._makeexpname(self.challengeprogramname, self.topic)
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')
		self.serverhost = 'serverNuc'
		self.sniffer = None
		self.run_counter = 0
		self.folder_name = '/data/{}/'.format(self.expname)
		if not os.path.exists(self.folder_name):
			os.makedirs(self.folder_name)

	def launchcontainers(self):
		# Launch the server container
		self.server = StacContainer(self.challengeprogramname, self.serverhost)

	def startserver(self):
		# Start the server process inside the server container
		self.server.execbashcmd('cd /home/challenge_program && ./startServer.sh', detach=True)
		# Wait a bit until the server is ready to use
		time.sleep(10)

	def startsniffer(self):
		print("Starting sniffer")
		self.sniffer = Sniffer(self.ports, showpackets=False)
		self.sniffer.start()

	def stopsniffer(self):
		self.sniffer.stop()

	def export_output(self):
		pass

	@classmethod
	def inputparser(self, inp):
		num_nodes = inp.getnumberofcities()
		num_edges = inp.getnumberofconnections()
		nodes = list(inp.cities)
		edges = list(inp.connections)
		input_str = "{}\n".format(num_nodes)
		
		for n in nodes:
			input_str = input_str + n + '\n'
		input_str = input_str + "{}\n".format(num_edges)
		for e in edges:
			e_str = "{} {} {} {} {} {} {} {}".format(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7])
			input_str = input_str + e_str + '\n'

		return input_str

	def run_inputs(self, inputs, num_samples):
		# Don't show warnings when ignoring SSL certificates
		requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

		#Convert inputs to a tuple with map name, input map string and secret value
		routemaps = [(genmap.random_word(12, 12), self.inputparser(i), i.getnumberofcities()) for i in inputs]

		time_started = time.time()

		# We now iterate over all maps for each sample number
		# (as opposed to sampling the same map many times repeatedly).
		for samplenumber in range(num_samples):

			for routemap_name_base, routemap_txt, number_of_airports in routemaps:

				# Append sample number to map name because we can't have >1 with same name.
				routemap_name = routemap_name_base + '_%03d' % samplenumber

				session = requests.Session()
				session.verify = False
				airplan_client = AirPlanClient(self.serverhost, self.ports[0], 'usr', 'pwd')

				self.sniffer.newinteraction(str(number_of_airports))

				self.sniffer.newphase('Login')		
				#print("Logging in to %s:%d" % (self.serverhost, self.ports[0]))
				airplan_client.login(session)
				time.sleep(0.1)

				self.sniffer.newphase('UploadMap')
				#print("Uploading map under the name %s" % (routemap_name))
				#print("Size: " + str(len(routemap_txt)))
				response = airplan_client.uploadRouteMapFromString(session, routemap_txt, 'txt', routemap_name)
				assert response.ok, "Could not upload map"
				assert 'Airplan! - Map Properties' in response.text, "Unexpected response"
				time.sleep(0.1)

				if False:
					self.sniffer.newphase('GetPropertiesPage')
					#print("Obtaining properties page")
					response = airplan_client.properties(session, "Cost")
					assert response.ok, "Could not fetch properties page"
					assert 'These properties are related to' in response.text, "Unexpected response"
					time.sleep(0.2)

				self.sniffer.newphase('GetMatrixPage')
				#print("Obtaining passenger capacity matrix page")
				response = airplan_client.passengerCapacity(session, airplan_client.propertiesURL)
				assert response.ok, "Could not fetch properties page:\n" + str(response)
				assert 'Passenger capacity between airports.' in response.text, "Unexpected response"
				time.sleep(0.1)

				# Check whether we need to delete the map
				if DELETE_UPLOADED_MAP_AFTER_EACH_ITERATION:
					self.sniffer.newphase('DeleteMap')
					print("Deleting map")
					response = airplan_client.delete_first_map(session)
					time.sleep(0.1)

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
		print("Saving " + self.tssfilename)
		self.sniffer.export_tss(self.folder_name + self.tssfilename)
		
		return self.sniffer.interactions

	def removecontainers(self):
		# Kill and remove the server container
		self.server.killrm()

	def killrmcontainers(self):
		self.server.killrm()

	def postfinishexperiment(self):
		# Do some magic analysis here...
		pass


