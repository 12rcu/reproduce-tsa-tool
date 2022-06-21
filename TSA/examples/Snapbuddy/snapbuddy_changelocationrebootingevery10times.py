#
# We try to keep the container alive here, to avoid aving to waste
# a 16-second delay on every single interaction.
#
# However, the server won't let you change locations more than 10
# times! On the 11th attempt you get an error. So we reset the
# container every 10 location changes.
#

import time, random, itertools

import docker
import requests
import tsa

from snapbuddy_client import SnapCaller

TOPIC = 'changelocationrebootingevery10times'
QUESTION = 'e2q31'
CHALLENGE = 'snapbuddy_1'
SERVER_HOST = 'server'
SERVER_PORT = 8080
USERNAME = 'devenmartinez@hotmail.com'
PASSWORD = 'PS1Ljv4NPs'
USERPUBLICKEY = '1234567'
IGNORE_IMAGES = True
NUM_SECRETS = 0 #To be updated after file read.
NUM_SAMPLES_PER_SECRET = 10

# Read the list of BSSIDS from a text file.
# Each element of the list is a comma-separated line.
#BSSIDS = open('bssids.txt', 'r').read().strip().split('\n')

city2bssids = {}

with open('cities', 'r') as citiesfile:
	for line in citiesfile:
		line = line.strip()
		if '","' in line:
			name, bssids = line.split('","')
			name = name.strip('"')
			bssids = bssids.strip('"')
			city2bssids[name] = bssids

NUM_SECRETS = len(city2bssids)

imagename = 'localhost:5000/stac_%s:v1.0' % CHALLENGE
dockerenv = {'DOCKER_HOST': 'tcp://%s:4243' % SERVER_HOST}
dockerclient = docker.from_env(version='auto', environment=dockerenv)

# Don't show warnings when ignoring SSL certificates
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

time_started = time.time()

s = sniffer.Sniffer([SERVER_PORT])
s.start()

# Start container
launchcmd = "bash -c 'cd challenge_program && ./start.sh'"
print("\nLaunching Docker container for %s" % CHALLENGE)
print("launchcmd is: %s" % launchcmd)
container = dockerclient.containers.run(imagename, launchcmd, network_mode='host', detach=True)
container = dockerclient.containers.get(container.id)
print("Launched container %s\n" % container.short_id)
time.sleep(16)


ith_request = 0

for j in range(NUM_SAMPLES_PER_SECRET):

	for cityname, bssids in city2bssids.iteritems():

		# Reset container every 10 iterations.
		# I think there is a Docker "restart" command for this. PEND: Look into it!
		if ith_request == 10:
			# Kill the container
			time.sleep(0.25)
			print("\nKilling container...")
			container.kill()
			print("Removing container...")
			container.remove()
			print("Killed and removed!\n")
			time.sleep(0.25)
			# Now restart it
			launchcmd = "bash -c 'cd challenge_program && ./start.sh'"
			print("\nLaunching Docker container for %s" % CHALLENGE)
			print("launchcmd is: %s" % launchcmd)
			container = dockerclient.containers.run(imagename, launchcmd, network_mode='host', detach=True)
			container = dockerclient.containers.get(container.id)
			print("Launched container %s\n" % container.short_id)
			time.sleep(16)
			# And restart the counter
			ith_request = 0

		secret = cityname
		s.newinteraction(secret)

		caller = SnapCaller(SERVER_HOST, SERVER_PORT, USERNAME, PASSWORD, IGNORE_IMAGES)
		response = caller.process(bssids, USERPUBLICKEY)

		ith_request += 1

		time.sleep(1)

# Stop the container
time.sleep(0.25)
print("\nKilling container...")
container.kill()
print("Removing container...")
container.remove()
print("Killed and removed!\n")
time.sleep(0.25)

# Stop sniffer
s.stop()

pcapfilename = '%s-%s-%s-%03dx%03d.pcap' % \
			(TOPIC, QUESTION, CHALLENGE, NUM_SECRETS, NUM_SAMPLES_PER_SECRET)
s.export(pcapfilename)

time_finished = time.time()
time_elapsed = time_finished - time_started
assert pcapfilename.endswith('.pcap')
timefilename = pcapfilename[:-5] + '.time'
print("Exporting time to: " + timefilename)
with open(timefilename, 'w') as timefile:
	timefile.write('%.2f\n' % time_elapsed)
