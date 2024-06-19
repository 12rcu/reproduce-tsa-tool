import time, os

import requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

from Medpedia.medpedia_client import MedpediaClient

from tsa import Sniffer
from tsa.new_stac import StacPlatform, StacContainer, App

MSEC_DELAY_BETWEEN_ARTICLES = 100
SERVER_HOST = 'serverNuc'
#SERVER_HOST = 'localhost'
SERVER_PORT = 8443

class MedpediaApp(App):

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
		#self.serverhost = 'serverNuc'
		self.serverhost = 'localhost'
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
		cmd = 'cd /home/challenge_program && ./startServer.sh'
		self.server.execbashcmd(cmd, detach=True)
		# Wait a bit until the server is ready to use
		time.sleep(25)

	def startsniffer(self):
		print("Starting sniffer")
		self.sniffer = Sniffer(self.ports, showpackets=False)
		self.sniffer.start()

	def stopsniffer(self):
		self.sniffer.stop()

	def export_output(self):
		pass

	def run_inputs(self, inputs, num_samples=16):

		time_started = time.time()

		c = MedpediaClient(self.serverhost, self.ports[0])

		# Iterate over all instances for each sample number
		for samplenumber in range(num_samples):
			for instance in inputs:
				articlename = instance.article
				secret = articlename
				#print("Starting interaction with secret=%s" % secret)
				self.sniffer.newinteraction(secret)
				# First we get the complete list (apparently this makes a difference!)
				#s.newphase('GetListAll')
				#allarticlenames = c.get_listall()
				self.sniffer.newphase('GetArticleHtml')
				response = c.get_article_response(articlename)
				html = response.content
				#size_html = len(html) if response.status_code is 200 else -1
				#size_padding = len(response.headers.get('X-Padding')) if 'X-Padding' in response.headers else -1
				#size_headers = sum([ len(k + ': ' + v + '\n') for (k, v) in response.headers.iteritems() ])
				#print("ARTICLE  %10d  %10d  %10d  %10d  %s" % (response.status_code, size_html, size_padding, size_headers, articlename))
				# Sleep a bit
				time.sleep(MSEC_DELAY_BETWEEN_ARTICLES * 0.001)

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

