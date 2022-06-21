import requests

# Don't show warnings when ignoring SSL certifs
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

from bs4 import BeautifulSoup as BS


class MedpediaClient(object):
	
	def __init__(self, serverhost='serverNuc', serverport=8443):
		self.serverhost = serverhost
		self.serverport = serverport
		self.session = requests.Session()
		self.session.verify = False
		self.baseurl = 'https://%s:%d/' % (self.serverhost, self.serverport)
		self.listall = None

	def log(self, *args):
		print(args)

	def makeurl(self, afterslash):
		return self.baseurl + afterslash

	def get_listall_from_file(self, pathname='medpedia_listAll.txt'):
		with open(pathname, 'r') as f:
			allarticlenames = f.read().strip().splitlines()
		return allarticlenames

	def get_listall(self):
		url = self.makeurl(afterslash='_listAll')
		response = self.session.get(url, allow_redirects=True)
		assert response.status_code is 200
		allarticlenames = response.content.strip().splitlines()
		return allarticlenames

	def get_article_response(self, articlename):
		assert articlename.startswith('A/')
		url = self.makeurl(afterslash=articlename)
		response = self.session.get(url, allow_redirects=True)
		content = response.content if response.status_code is 200 else None
		return response

	def get_article(self, articlename):
		response = self.get_article_response(articlename)
		return response.status_code, response.content

	def extract_imagenames(self, html):
		soup = BS(html, "lxml")
		imgtags = soup.findAll('img')
		srcs = [img['src'] for img in imgtags]
		assert all(src.startswith('../I') for src in srcs)
		return [src[3:] for src in srcs]

	def get_image(self, imagename):
		assert imagename.startswith('I/')
		url = self.makeurl(afterslash=imagename)
		response = self.session.get(url, allow_redirects=True)
		content = response.content if response.status_code is 200 else None
		return response.status_code, response.content

	def search(self, prefix):
		url = self.makeurl(afterslash='titles?prefix=%s' % prefix)
		response = self.session.get(url, allow_redirects=True)
		content = response.content if response.status_code is 200 else None
		return response.status_code, response.content



