import time, os
from collections import Counter

import requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

from .railyard import RailyardInput

from tsa import sniffer
from tsa.new_stac import StacPlatform, StacContainer, App


PASSWORD = "zaptrap"


class RailyardClient(object):
	"""
	Provides a simple abstraction layer around the necessary requests and protocol.
	The session is provided by the user so that authentication is optional.
	Example usage:
	
	session = requests.Session()
	session.verify = False
	client = RailyardClient(session, 'serverNuc')
	client.authenticate("mypassword")
	client.addcar("Coal Car", "my little car")
	...

	If the session already has a valid cookie, authentication is not necessary.
	"""

	def __init__(self, session, serverhost, port=4567):
		self.session = session
		self.serverhost = serverhost
		self.port = port

	def mkurl(self, path):
		assert path.startswith('/')
		return 'https://{}:{}/api/v1{}'.format(self.serverhost, self.port, path)

	def authenticate(self, password):
		url = self.mkurl('/authenticate.json')
		json = {"password": password}
		response = self.session.post(url, json=json)
		assert response.ok, "Expected OK but got: {} content: {}".format(response.status_code, response.content)
		return response

	def addcar(self, car_type, car_identifier):
		url = self.mkurl('/platforms/a/add_car.json')
		json = {"car_type": car_type, "car_identifier": car_identifier}
		response = self.session.post(url, json=json)
		assert response.ok, "Expected OK but got: {} content: {}".format(response.status_code, response.content)
		return response

	def addcargo(self, material, quantity, description):
		url = self.mkurl('/platforms/a/cargo/add.json')
		json = {"description": description, "quantity": quantity, "material": material}
		response = self.session.post(url, json=json)
		assert response.ok, "Expected OK but got: {} content: {}".format(response.status_code, response.content)
		return response

	def addpersonnel(self, name):
		url = self.mkurl('/platforms/a/personnel/add.json')
		json = {"name": name}
		response = self.session.post(url, json=json)
		assert response.ok, "Expected OK but got: {} content: {}".format(response.status_code, response.content)
		return response

	def setschedule(self, schedule):
		url = self.mkurl('/platforms/a/schedule.json')
		stops = dict((str(stoptime), stopname) for (stoptime, stopname) in schedule.iteritems())
		response = self.session.patch(url, json={"departs": 1, "stops": stops})
		assert response.ok, "Expected OK but got: {} content: {}".format(response.status_code, response.content)
		return response

	def sendout(self):
		url = self.mkurl('/platforms/a/send_out.json')
		response = self.session.post(url)
		assert response.ok, "Expected OK but got: {} content: {}".format(response.status_code, response.content)
		return response


class RailyardApp(App):

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
		#print(self._makebanner("EXPERIMENT: " + self.expname, self.params)
		self.serverhost = 'serverNuc'
		self.ports = ports
		self.sniffer = None
		self.run_counter = 0

		self.folder_name = '/data/{}/'.format(self.expname)
		if not os.path.exists(self.folder_name):
			os.makedirs(self.folder_name)

	def launchcontainers(self):
		# Remove any non-registry containers on all hosts
		#StacPlatform.cleanuphosts(['clientNuc', 'serverNuc', 'masterNuc'])
		# Launch the server container
		self.server = StacContainer(self.challengeprogramname, self.serverhost)

	def startserver(self):
		# Start the server process inside the server container
		cmd = 'cd /home/challenge_program && ./startServer.sh {}'.format(PASSWORD)
		self.server.execbashcmd(cmd, detach=True)
		# Wait a bit until the server is ready to use
		time.sleep(3)

	def startsniffer(self):
		print("Starting sniffer")
		self.sniffer = sniffer.Sniffer(self.ports, showpackets=False)
		self.sniffer.start()

	def stopsniffer(self):
		self.sniffer.stop()

	def export_output(self):
		pass

	def run_inputs(self, inputs, num_samples):

		time_started = time.time()

		# Create the client and authenticate its session only once.
		# We do this inside a "fake" interaction, since we aren't interested in sniffing it.
		self.sniffer.newinteraction("X")
		self.sniffer.newphase("X")
		self.session = requests.Session()
		self.session.verify = False
		self.client = RailyardClient(self.session, self.serverhost)
		r = self.client.authenticate(PASSWORD)

		# Iterate over all instances for each sample number
		for samplenumber in range(num_samples):

			for instance in inputs:

				# Set up the server state to reflect the instance
				# We do this as a "fake" interaction, for later deletion.
				self.sniffer.newinteraction("X")
				self.sniffer.newphase("X")

				for (cartype, carids) in instance.cars.iteritems():
					for carid in carids:
						r = self.client.addcar(cartype, carid)
	
				for (material, descriptions) in instance.cargo.iteritems():
					for description in descriptions:
						r = self.client.addcargo(material, 1, description)

				for name in instance.personnel:
					r = self.client.addpersonnel(name)

				r = self.client.setschedule(instance.schedule)
				# DEBUGGING
				#print(instance)
				#print(r.content)

				# Now we actually send out the train.
				# This is the part that we want to sniff, so we start a real interaction here.
				time.sleep(0.01)
				self.sniffer.newinteraction(instance.getsecret())
				self.sniffer.newphase("SendOutTrain")
				time.sleep(0.01)	
				r = self.client.sendout()
				time.sleep(0.01)

				# DEBUG
				#import json
				#z = json.loads(r.content)
				#print(z['train'])
				#nbytes = len(z['train'])
				#print("\n\nSEC2OBS {} {}\n".format(instance.getsecret(), nbytes))


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
		
		self.sniffer.clean_marked_interactions()
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

	def postfinishexperiment(self):
		# Do some magic analysis here...
		pass

