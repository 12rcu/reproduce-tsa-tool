import time, random, docker, os

import requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

from tsa import Sniffer
from tsa.new_stac import StacPlatform, StacContainer, App

USERBIGNUM_MINDIGITS = 256
USERBIGNUM_MAXDIGITS = 1024

def gen_userbignum():
	numdigits = random.randint(USERBIGNUM_MINDIGITS, USERBIGNUM_MAXDIGITS)
	digits = [random.choice('0123456789') for _ in range(numdigits)]
	digitstr = ''.join(digits)
	return int(digitstr)

class GabfeedApp(App):

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

		self.user_bignum_str = str(gen_userbignum())

		self.folder_name = '/data/{}/'.format(self.expname)
		if not os.path.exists(self.folder_name):
			os.makedirs(self.folder_name)

	def launchcontainers(self):
		# Remove any non-registry containers on all hosts
		StacPlatform.cleanuphosts(['clientNuc', 'serverNuc', 'masterNuc'])
		# Launch the server container
		#self.server = StacContainer(self.challengeprogramname, self.serverhost)

	def startserver(self):
		pass
		# Start the server process inside the server container
		#cmd = 'cd /home/challenge_program && ./startServer.sh {}'.format(PASSWORD)
		#self.server.execbashcmd(cmd, detach=True)
		# Wait a bit until the server is ready to use
		#time.sleep(3)

	def startsniffer(self):
		print("Starting sniffer")
		self.sniffer = Sniffer(self.ports, showpackets=False)
		self.sniffer.start()

	def stopsniffer(self):
		self.sniffer.stop()

	def export_output(self):
		pass

	def run_inputs(self, inputs, num_samples):
		#TODO This is just original script imported here, we may need to convert dockerclient stuff to StacPlatform abstract version.
		time_started = time.time()

		imagename = 'localhost:5000/stac_%s:v1.0' % self.challengeprogramname
		dockerenv = {'DOCKER_HOST': 'tcp://%s:4243' % self.serverhost}
		dockerclient = docker.from_env(version='auto', environment=dockerenv)

		# Iterate over all instances for each sample number
		for samplenumber in range(num_samples):
			for instance in inputs:

				bitstring = instance.getinput()
				decimalvalue = int(bitstring, 2)
				server_privatekey = '0x%0.16x' % decimalvalue

				secret = instance.getsecret()

				launchcmd = "bash -c 'cd challenge_program && echo -n %s > ServersPrivateKey.txt && ./start.sh'" % server_privatekey

				#print("Launching Docker container for %s" % self.challengeprogramname)
				#print("launchcmd is: %s" % launchcmd)
				container = dockerclient.containers.run(imagename, launchcmd, network_mode='host', detach=True)
				container = dockerclient.containers.get(container.id)
				#print("Launched container %s" % container.short_id)
				time.sleep(2)

				self.sniffer.newinteraction(secret)

				url = 'https://%s:%d/authenticate' % (self.serverhost, self.ports[0])
				#print("Posting to authentication URL: " + self.user_bignum_str)
				response = requests.post(url, files={'A': self.user_bignum_str}, verify=False)
				assert response.ok
				#print("OK!\n\n" + response.content + "\n\n")

				#print("Killing container...")
				container.kill()
				container.remove()
				#print("Killed!")

				time.sleep(0.5)

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
		#print("Saving " + self.tssfilename)
		#self.sniffer.export_tss(self.folder_name + self.tssfilename)
		
		return self.sniffer.interactions

	def removecontainers(self):
		pass
		# Kill and remove the server container
		#self.server.killrm()

	def postfinishexperiment(self):
		# Do some magic analysis here...
		pass

