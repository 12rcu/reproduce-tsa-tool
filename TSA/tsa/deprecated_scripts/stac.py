import time, datetime, re

import docker

import requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

from tsa.sniffer import Sniffer


class StacPlatform(object):
	allnodes = 'serverNuc', 'clientNuc', 'masterNuc'

	@staticmethod
	def makedockerhost(dockerhostname, dockerport):
		return 'tcp://%s:%s' % (dockerhostname, str(dockerport))

	@staticmethod
	def dockercontainertoimagename(cnt):
		config = cnt.attrs.get('Config')
		# including host, port, version, and image name
		fullimagename = config.get('Image')
		REGEX = r'([^/]+)/([^:]+):v([0-9.]+)'
		match = re.match(REGEX, fullimagename)
		# discard host, port, and version
		shortimagename = match.group(2)
		return shortimagename

	@staticmethod
	def cleanuphost(dockerhostname, dockerport=4243):
		'''
		Kill and remove ALL containers except registry on the given host.
		'''
		dockerhost = StacPlatform.makedockerhost(dockerhostname, dockerport)
		dockerenv = {'DOCKER_HOST': dockerhost}
		dockerclient = docker.from_env(version='auto', environment=dockerenv)
		dockercontainers = dockerclient.containers.list()
		for dockercontainer in dockercontainers:
			shortid = dockercontainer.short_id
			shortimagename = StacPlatform.dockercontainertoimagename(dockercontainer)
			if shortimagename != 'registry':
				print("Killing and removing container %s (%s) on %s" % (shortid,
					shortimagename, dockerhostname))
				dockercontainer.kill()
				dockercontainer.remove()
			else:
				print("Preserving container %s (%s) on %s" % (shortid,
					shortimagename, dockerhostname))

	@staticmethod
	def cleanuphosts(dockerhostnames, dockerport=4243):
		'''
		Kill and remove ALL containers except registry on the given hosts.
		This assumes that all the hosts use the same port for Docker API.
		'''
		for hostname in dockerhostnames:
			StacPlatform.cleanuphost(hostname, dockerport)

	@staticmethod
	def cleanup():
		'''
		Kill and remove ALL containers except registry on all STAC Plaform nodes.
		This is convenient to ensure a clean working environment before starting
		an experiment, but beware -- it's easy to ruin someone else's experiment
		by running this! Make sure to be in exclusive posession of the platform.
		'''
		StacPlatform.cleanuphosts(StacPlatform.allnodes)




class StacContainer(object):

	def __init__(self, challengeprogramname, dockerhost):
		self.challengeprogramname = challengeprogramname
		self.imagename = self._makeimagename(self.challengeprogramname)
		self.dockerhostname = dockerhost
		self.dockerport = 4243  # hardcoded for now, this is the standard Docker API port
		self.dockerhost = StacPlatform.makedockerhost(self.dockerhostname, self.dockerport)
		# Create environment to use Docker API
		self.dockerenv = {'DOCKER_HOST': self.dockerhost}
		# Create Docker client
		self.dockerclient = docker.from_env(version='auto', environment=self.dockerenv)
		# Create a command to sleep a whole year. This lets us exec commands until
		# we're done, and then we kill the container. Dirty, but works. Cleaner
		# alternatives seem to have other limitations...
		self.launchcmd = "bash -c 'sleep 365d'"
		# Create the container and launch it with the above "fake" command.
		# Network mode 'host' means all networking uses the host interface.
		# Detach means this call is nonblocking (container runs in background).
		print("Starting container for %s on %s" % (self.challengeprogramname, self.dockerhost))
		self.container = self.dockerclient.containers.run(self.imagename,
			self.launchcmd,
			network_mode='host',
			detach=True
			)
		# Now reobtain the container object after having done the launch.
		# Doing this seemed to fix an odd problem long ago (?)
		self.container = self.dockerclient.containers.get(self.container.id)
		print("Started container %s" % self.container.short_id)

	def _makeimagename(self, challengeprogramname):
		return 'localhost:5000/stac_%s:v1.0' % challengeprogramname

	def putsinglefile(self, localpathname, remotedestdir, remotefilename):
		'''
		This is convenient because the API does not offer a simple way to
		inject a single file into a running container (only a tar archive).
		'''
		# Make an in-memory tarfile that contains the given file
		import tarfile, StringIO
		tar_flo = StringIO.StringIO()
		tar = tarfile.open(mode='w', fileobj=tar_flo)
		with open(localpathname, 'r') as localfile:
			data = localfile.read()
			flo = StringIO.StringIO(data)
			info = tarfile.TarInfo(name=remotefilename)
			info.size = len(flo.buf)
			info.mtime = time.time()
			tar.addfile(tarinfo=info, fileobj=flo)
		tar.close()
		tardata = tar_flo.getvalue()
		# Copy its contents to the container
		print("Copying file %s into container %s in directory %s as %s" %
			(localpathname, self.container.short_id, remotedestdir, remotefilename))
		success = self.container.put_archive(remotedestdir, tardata)
		assert success

	def kill(self):
		return self.container.kill()

	def remove(self):
		return self.container.remove()

	def killrm(self):
		print("Killing and removing container %s (%s) on %s" %
			(self.container.short_id,
			self.challengeprogramname,
			self.dockerhostname))
		self.container.kill()
		return self.container.remove()

	def fufi(self, bashcommand, **kwargs):
		print(bashcommand)
		print(kwargs)

	def execbashcmd(self, bashcommand, **kwargs):
		return self.container.exec_run("bash -c '%s'" % bashcommand, **kwargs)


class ChallengeProgramDriver(object):

	def __init__(self, challengeprogramname, topic, ports, params, labelwith):
		self.challengeprogramname = challengeprogramname
		self.topic = topic
		self.ports = ports
		self.params = params
		self.params['labelwith'] = labelwith
		self.labelwith = labelwith
		self.exptime = time.time()
		self.exptimestamp = self._maketimestamp()
		self.expname = self._makeexpname(self.challengeprogramname, self.topic, self.params, self.labelwith)
		self.pcapfilename = self._makeoutputfilename('.pcap')
		self.tssfilename = self._makeoutputfilename('.tss')
		print(self._makebanner("EXPERIMENT: " + self.expname, self.params))

	def run(self):
		methodnamesinorder = [
			'setparameters',
			'startexperiment',
			'prelaunchcontainers',
			'launchcontainers',
			'postlaunchcontainers',
			'prestartservers',
			'startservers',
			'poststartservers',
			'createsniffer',
			'prestartsniffer',
			'startsniffer',
			'poststartsniffer',
			'preinteractions',
			'interactions',
			'postinteractions',
			'prestopsniffer',
			'stopsniffer',
			'poststopsniffer',
			'preremovecontainers',
			'removecontainers',
			'postremovecontainers',
			'prefinishexperiment',
			'finishexperiment',
			'postfinishexperiment',
			]
		for methodname in methodnamesinorder:
			if hasattr(self, methodname):
				method = getattr(self, methodname)
				print(self._makebanner(methodname))
				method()

	def _makeexpname(self, challengeprogramname, topic, params, labelwith):
		if labelwith:
			relevant_values_in_order = [params[paramname] for paramname in labelwith]
			paramstr = '_'.join([str(v) for v in relevant_values_in_order])
		else:
			paramstr = 'NoParams'
		timestamp = self.exptimestamp
		expname = '-'.join([challengeprogramname, topic, paramstr, timestamp])
		return expname

	def _maketimestamp(self, fmtstr='%Y%m%d%H%M%S'):
		return datetime.datetime.today().strftime(fmtstr)

	def _makeoutputfilename(self, extension):
		assert extension.startswith('.')
		return self.expname + extension

	def _makebanner(self, title, params=None, widthchars=100):
		solidline = widthchars * '='
		titleline = title.center(widthchars)
		alllines = [solidline, '', titleline, '']
		if params:
			otherlines = [('    %s: %s' % (str(k), str(v))) for (k, v) in params.iteritems()]
			alllines.extend(otherlines)
			alllines.append('')
		alllines.append(solidline)
		block = '\n'.join(alllines)
		return '\n' + block + '\n'

	def saveoutput(self, content, extension):
		outputfilename = self._makeoutputfilename(extension)
		with open(outputfilename, 'w') as outputfile:
			print("Saving " + outputfilename)
			outputfile.write(content)

	# Default method implementations

	def createsniffer(self):
		self.sniffer = Sniffer(self.ports)

	def startsniffer(self):
		self.sniffer.start()
		time.sleep(0.25)

	def stopsniffer(self):
		time.sleep(0.25)
		self.sniffer.stop()

	def finishexperiment(self):
		self.endtime = time.time()
		self.elapsed = self.endtime - self.exptime
		print("\n\nFinishing experiment %s after %.0f seconds\n" % (self.expname, self.elapsed))
		# save time file
		self.saveoutput('%.2f\n' % self.elapsed, '.time')
		self.sniffer.stop()
		# save params file
		self.saveoutput(str(self.params) + '\n', '.params')
		# save pcap file
		print("Saving " + self.pcapfilename)
		self.sniffer.export(self.pcapfilename)
		# save tss file
		print("Saving " + self.tssfilename)
		self.sniffer.export_tss(self.tssfilename)		





'''
d = ChallengeProgramDriver('simplevote_1', 'TestBallot', [('foo', 35), ('baz', 'MEH')])

StacPlatform.cleanup()


server = StacContainer('simplevote_1', 'serverNuc')
client = StacContainer('simplevote_1', 'clientNuc')

server.execbashcmd('cd /home/examples && ./start.sh', detach=True)
time.sleep(3)

client.putsinglefile('assets/interact.py', '/home/examples', 'interact.py')
client.putsinglefile('assets/updateballot.py', '/home/examples', 'updateballot.py')

registrationkey = "032113400906"

client.execbashcmd('cd /home/examples && python updateballot.py serverNuc 8443 KathyRBown@rhyta.com Iapeequu3dah %s' % registrationkey,
	stdout=True, detach=False)

'''


