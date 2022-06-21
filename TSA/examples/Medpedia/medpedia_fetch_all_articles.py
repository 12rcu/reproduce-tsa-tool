import time
import docker
import random

from tsa import Sniffer

from medpedia_client import MedpediaClient

TOPIC = 'whicharticle'
SERVER_HOST = 'serverNuc'
SERVER_PORT = 8443
NUMBER_OF_ARTICLES_TO_FETCH = 4  # Use 0 for "fetch all", >0 for a random sample
NUMBER_OF_TIMES_PER_ARTICLE = 3
MSEC_DELAY_BETWEEN_ARTICLES = 100

question, challenge = ('e5q09', 'medpedia')

imagename = 'localhost:5000/stac_%s:v1.0' % challenge
dockerenv = {'DOCKER_HOST': 'tcp://%s:4243' % SERVER_HOST}
dockerclient = docker.from_env(version='auto', environment=dockerenv)

launchcmd = "bash -c 'cd challenge_program && ./startServer.sh'"
print("Launching Docker container for %s on %s" % (question, challenge))
print("launchcmd is: %s" % launchcmd)

container = dockerclient.containers.run(imagename, launchcmd, network_mode='host', detach=True)
container = dockerclient.containers.get(container.id)

print("Launching container %s" % container.short_id)

# It takes about 21 seconds for the server to boot. Wait 25 to be on the safe side.
time.sleep(25)

print("Server should be ready by now.")
print("")

c = MedpediaClient(SERVER_HOST, SERVER_PORT)

allarticlenames = c.get_listall()

time_started = time.time()

print("Starting sniffer")
s = Sniffer([SERVER_PORT])
s.start()

# Build the list of articles to fetch
if NUMBER_OF_ARTICLES_TO_FETCH is 0:
	articlenames = allarticlenames
else:
	articlenames = random.sample(allarticlenames, NUMBER_OF_ARTICLES_TO_FETCH)

# Now fetch the articles. Some of them could yield 404; handle that gracefully, too.
for articlename in articlenames:
	for i in range(NUMBER_OF_TIMES_PER_ARTICLE):
		# Record the start of interaction
		secret = articlename
		print("\n\nStarting interaction with secret=%s" % secret)
		s.newinteraction(secret)
		# First we get the complete list (apparently this makes a difference!)
		#s.newphase('GetListAll')
		#allarticlenames = c.get_listall()
		s.newphase('GetArticleHtml')
		response = c.get_article_response(articlename)
		html = response.content
		size_html = len(html) if response.status_code is 200 else -1
		size_padding = len(response.headers.get('X-Padding')) if 'X-Padding' in response.headers else -1
		size_headers = sum([ len(k + ': ' + v + '\n') for (k, v) in response.headers.iteritems() ])
		print("ARTICLE  %10d  %10d  %10d  %10d  %s" % (response.status_code, size_html, size_padding, size_headers, articlename))
		# Sleep a bit
		time.sleep(MSEC_DELAY_BETWEEN_ARTICLES * 0.001)


print("")
time.sleep(1)

print("Killing and removing container...")
container.kill()
print("Killed!")
container.remove()
print("Removed!")


print("Stopping sniffer")
s.stop()

# Take a look at the side-channel sizes
scsizelists = sniffer.Transform.rd_aggregate_phase(s.interactions)
print('\nSide-channel sizes:')
print('\n'.join(map(str, scsizelists)))


number_of_articles_actually_fetched = len(articlenames)

pcapfilename = '%s-%s-%s-%05dx%03d.pcap' % \
	(TOPIC, question, challenge, number_of_articles_actually_fetched, NUMBER_OF_TIMES_PER_ARTICLE)
print("Exporting pcap to: " + pcapfilename)
s.export(pcapfilename)

time_finished = time.time()
time_elapsed = time_finished - time_started
assert pcapfilename.endswith('.pcap')
timefilename = pcapfilename[:-5] + '.time'
print("Exporting time to: " + timefilename)
with open(timefilename, 'w') as timefile:
	timefile.write('%.2f\n' % time_elapsed)
tssfilename = pcapfilename[:-5] + '.tss'
print("Exporting tss to: " + tssfilename)
s.export_tss(tssfilename)


