import requests
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

class RailyardClient(object):

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