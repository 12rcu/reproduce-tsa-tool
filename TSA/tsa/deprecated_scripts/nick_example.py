import time, random, docker
from collections import Counter

import requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

import sniffer
from new_stac import StacPlatform, StacContainer, App

class TestApp(App):

	def __init__(self, topic, ports):
		self.challengeprogramname = ''
		self.topic = topic
		self.ports = ports
		self.params = {}
		self.exptime = time.time()
		self.exptimestamp = self._maketimestamp()
		self.expname = self._makeexpname(self.challengeprogramname, self.topic)
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')
		#print(self._makebanner("EXPERIMENT: " + self.expname, self.params)
		self.sniffer = None
		self.run_counter = 0

	def startsniffer(self):
		print("Starting sniffer")
		self.sniffer = sniffer.Sniffer(self.ports, showpackets=False)
		self.sniffer.start()

	def stopsniffer(self):
		self.sniffer.stop()

	def export_output(self):
		pass

	def run_inputs(self, inputs, num_samples):
		#TODO This is just original script imported here, we may need to convert dockerclient stuff to StacPlatform abstract version.
		time_started = time.time()


		url = "http://www.thomas-bayer.com/restnames/name.groovy"

		# Iterate over all instances for each sample number
		for samplenumber in range(num_samples):
			for instance in inputs:

				self.sniffer.newinteraction(instance)
				response = requests.get(url, params={'name': instance}, verify=False)
				assert response.ok
				print("OK!\n\n" + response.content + "\n\n")


				time.sleep(0.5)

	def finishexperiment(self):
		self.endtime = time.time()
		self.elapsed = self.endtime - self.exptime
		print("\n\nFinishing experiment %s after %.0f seconds\n" % (self.expname, self.elapsed))
		# save time file
		self.saveoutput('%.2f\n' % self.elapsed, '.time')
		self.sniffer.stop()
		#self.sniffer.clean_marked_interactions()

		self.exptimestamp = self._maketimestamp()
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')
		# save pcap file
		print("Saving " + self.pcapfilename)
		self.sniffer.export(self.pcapfilename)
		# save tss file
		print("Saving " + self.tssfilename)
		self.sniffer.export_tss(self.tssfilename)
		
		return self.sniffer.interactions

	def removecontainers(self):
		pass
		# Kill and remove the server container
		#self.server.killrm()

	def postfinishexperiment(self):
		# Do some magic analysis here...
		pass

	

if __name__ == '__main__':
	
	# This is just a little test of the App above.
	# This has no seeds. It just starts from empty instances
	# and applies a bunch of mutations to each empty instance.
	
	SAMPLES_PER_INSTANCE = 2
	instances = ['Thomas', 'Equinox', 'Titus', 'Androvicus', 'Boris', 'Nick', 'Hans', 'Helga']

	app = TestApp('firsttest', [80, 443])
	app.startsniffer()
	app.run_inputs(instances, SAMPLES_PER_INSTANCE)
	app.stopsniffer()
	app.finishexperiment()

